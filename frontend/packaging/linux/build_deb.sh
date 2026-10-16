#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_ROOT="${SCRIPT_DIR}/build/autoscene-ui"
OUTPUT_DIR="${PROJECT_DIR}/dist-linux"

rm -rf "${BUILD_ROOT}"
mkdir -p \
    "${BUILD_ROOT}/opt/autoscene-ui/autoscene_gui" \
    "${BUILD_ROOT}/opt/autoscene-ui/vendor" \
    "${BUILD_ROOT}/usr/share/doc/autoscene-ui" \
    "${OUTPUT_DIR}"

cp -a "${SCRIPT_DIR}/DEBIAN" "${BUILD_ROOT}/"
cp -a "${SCRIPT_DIR}/usr" "${BUILD_ROOT}/"
cp "${PROJECT_DIR}/autoscene_ui.py" "${BUILD_ROOT}/opt/autoscene-ui/"
cp "${PROJECT_DIR}/autoscene_gui/"*.py "${BUILD_ROOT}/opt/autoscene-ui/autoscene_gui/"
cp "${PROJECT_DIR}/README.md" "${BUILD_ROOT}/opt/autoscene-ui/"
cp -a "${SCRIPT_DIR}/vendor/websocket" "${BUILD_ROOT}/opt/autoscene-ui/vendor/"
cp "${SCRIPT_DIR}/vendor/licenses/websocket-client-copyright" \
    "${BUILD_ROOT}/usr/share/doc/autoscene-ui/"

# macOS archives may carry AppleDouble metadata into a Linux build directory.
# Debian control archives must contain only real package metadata files.
find "${BUILD_ROOT}" -type f \( -name '._*' -o -name '.DS_Store' \) -delete

find "${BUILD_ROOT}" -type d -exec chmod 0755 {} +
find "${BUILD_ROOT}" -type f -exec chmod 0644 {} +
chmod 0755 "${BUILD_ROOT}/usr/bin/autoscene-ui"

python3 -m compileall -q "${BUILD_ROOT}/opt/autoscene-ui"
find "${BUILD_ROOT}" -type d -name __pycache__ -prune -exec rm -rf {} +

PACKAGE_PATH="${OUTPUT_DIR}/autoscene-ui_1.1.2_all.deb"

if command -v dpkg-deb >/dev/null 2>&1; then
    dpkg-deb --build --root-owner-group "${BUILD_ROOT}" "${PACKAGE_PATH}"
    dpkg-deb --info "${PACKAGE_PATH}"
else
    # macOS 构建机没有 dpkg-deb；按 Debian binary package 2.0 标准生成 ar 包。
    # 所有归档成员显式写成 root:root，安装到 Ubuntu 后不会继承 macOS 用户 UID。
    PORTABLE_DIR="${SCRIPT_DIR}/build/portable-deb"
    rm -rf "${PORTABLE_DIR}"
    mkdir -p "${PORTABLE_DIR}"
    printf '2.0\n' > "${PORTABLE_DIR}/debian-binary"
    COPYFILE_DISABLE=1 tar --no-xattrs --uid 0 --gid 0 --uname root --gname root \
        -cf "${PORTABLE_DIR}/control.tar" -C "${BUILD_ROOT}/DEBIAN" .
    COPYFILE_DISABLE=1 tar --no-xattrs --uid 0 --gid 0 --uname root --gname root \
        -cf "${PORTABLE_DIR}/data.tar" -C "${BUILD_ROOT}" opt usr
    xz -9 -f "${PORTABLE_DIR}/control.tar"
    xz -9 -f "${PORTABLE_DIR}/data.tar"
    rm -f "${PACKAGE_PATH}"
    (
        cd "${PORTABLE_DIR}"
        # -S prevents BSD ar from adding a Mach-O symbol table member, which is
        # not part of the Debian package format.
        ar -rcS "${PACKAGE_PATH}" debian-binary control.tar.xz data.tar.xz
    )
    echo "Portable Debian archive members:"
    ar -t "${PACKAGE_PATH}"
fi

echo "Built: ${PACKAGE_PATH}"
