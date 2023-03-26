function [X,Y] = draw_car(x,y,theta,L,H)
X = [-L/2  L/2  L/2  -L/2];
Y = [-H/2  -H/2  H/2  H/2];

cth = cos(theta) ; 
sth = sin(theta);
xrot =  X*cth - Y*sth;
yrot =  X*sth + Y*cth;

X = x+xrot;
Y = y+yrot;

X = [X,X(1)];
Y = [Y,Y(1)];