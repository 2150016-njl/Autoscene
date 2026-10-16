#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
VERSION="1.1.2"
BUNDLE_NAME="UI-0820"
DELIVERY_DIR="${ROOT_DIR}/delivery"
BUILD_PARENT="${DELIVERY_DIR}/.build-${VERSION}"
STAGE_DIR="${BUILD_PARENT}/${BUNDLE_NAME}"
ARCHIVE_PATH="${DELIVERY_DIR}/${BUNDLE_NAME}.zip"
DEB_PATH="${PROJECT_DIR}/dist-linux/autoscene-ui_${VERSION}_all.deb"

"${PROJECT_DIR}/packaging/linux/build_deb.sh"

rm -rf "${BUILD_PARENT}"
mkdir -p "${STAGE_DIR}" "${DELIVERY_DIR}"

RSYNC_EXCLUDES=(
    --exclude '.DS_Store'
    --exclude '._*'
    --exclude '__pycache__'
    --exclude '*.pyc'
    --exclude '.backup_v1'
    --exclude 'dist-linux'
    --exclude 'packaging/linux/build'
    --exclude '完整交付操作指南.md'
    --exclude '交付包内容.md'
)

COPYFILE_DISABLE=1 rsync -a "${RSYNC_EXCLUDES[@]}" \
    "${ROOT_DIR}/AutoScene/" "${STAGE_DIR}/AutoScene/"
COPYFILE_DISABLE=1 rsync -a "${RSYNC_EXCLUDES[@]}" \
    "${PROJECT_DIR}/" "${STAGE_DIR}/frontend/"
COPYFILE_DISABLE=1 rsync -a "${RSYNC_EXCLUDES[@]}" \
    "${ROOT_DIR}/third_party/" "${STAGE_DIR}/third_party/"

cp "${DEB_PATH}" "${STAGE_DIR}/"
cp "${PROJECT_DIR}/完整交付操作指南.md" "${STAGE_DIR}/DEPLOYMENT_GUIDE_CN.md"
cp "${PROJECT_DIR}/交付包内容.md" "${STAGE_DIR}/PACKAGE_CONTENTS_CN.md"

find "${STAGE_DIR}" -type f \( -name '.DS_Store' -o -name '._*' -o -name '*.pyc' \) -delete
find "${STAGE_DIR}" -type d -name '__pycache__' -prune -exec rm -rf {} +
chmod +x "${STAGE_DIR}/frontend/"*.sh "${STAGE_DIR}/frontend/packaging/"*.sh

rm -f "${ARCHIVE_PATH}"
(
    cd "${BUILD_PARENT}"
    COPYFILE_DISABLE=1 zip -X -q -r "${ARCHIVE_PATH}" "${BUNDLE_NAME}"
)

unzip -tq "${ARCHIVE_PATH}"
shasum -a 256 "${ARCHIVE_PATH}"
du -h "${ARCHIVE_PATH}"
rm -rf "${BUILD_PARENT}"
echo "Built: ${ARCHIVE_PATH}"
