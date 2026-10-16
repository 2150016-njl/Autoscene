function [Global_X,Global_Y]= LatLon2XY(Lat,Lon)

% output parameters
% LAT           uint32     latitude of positioning        dd
% LON          uint32     longitude of positioning        dd

% Lat的整数部分有4位，这里除以100后向下取整，得到其前两位
% 即度的部分，分和秒在原分和秒的基础上除以60
LAT_rad = Lat*pi/180; % rad
LON_rad = Lon*pi/180; % rad
[Global_X,Global_Y] = GaussianMapDirect(LAT_rad,LON_rad);

function [x,y] = GaussianMapDirect(B,L)
% WGS84椭球参数：
% a 地球椭圆的长半轴  m
a = 6378137.00000000000000; 
% f 扁率
f = 1/298.257223563;
% b 短半轴 m
b = a*(1-f);
% e 第一偏心率
e = sqrt((a^2 - b^2)/a^2);
% ep 第二偏心率
ep = sqrt((a^2 - b^2)/b^2);
% 根据3度带确认中央子午线
% Num 带号
Num = fix((L*180/pi)/3);
% L0 中央子午线经度  rad
L0 = 3*Num*pi/180;
% l 经度差 rad
l = L-L0;
% N 卯酉圈的半径 m
N = a/sqrt(1-e^2*sin(B)^2);
% t
t = tan(B);
% eta
eta = ep*cos(B);
% 计算弧长
Y=Lat2Mer(B,a,e);
% 计算x,y
y = Y + N/2* sin(B)*cos(B)*l^2+...
      N/24*sin(B)*cos(B)^3*(5-t^2+9*eta^2+4*eta^4)*l^4+...
      N/720*sin(B)*cos(B)^5*(61-58*t^2+t^4+270*eta^2-330*eta^2*t^2)*l^6;
x = N * cos(B)*l + ...
    N/6*cos(B)^3*(1-t^2+eta^2)*l^3+...
    N/120*cos(B)^5*(5-18*t^2+t^4+14*eta^2-58*eta^2*t^2)*l^5;
% 保证y是正数
x = x + 500000;


function Y = Lat2Mer(B,a,e)
% 由纬度B计算对应的子午线弧长
% B rad
m0 = a*(1-e^2);
m2 = 3/2*e^2*m0;
m4 = 5/4*e^2*m2;
m6 = 7/6*e^2*m4;
m8 = 9/8*e^2*m6;

a0 = m0 + m2/2 +3/8*m4 + 5/16*m6 + 35/128*m8;
a2 = m2/2+m4/2 + 15/32*m6 + 7/16*m8;
a4 = m4/8 +3/16*m6 + 7/32 *m8;
a6 = m6/32 + m8/16;
a8 = m8/128;

Y = a0 * B - (a2*sin(2*B))/2 + (a4*sin(4*B))/4 - (a6*sin(6*B))/6 +(a8*sin(8*B))/8;
