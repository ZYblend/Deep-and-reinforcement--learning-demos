%% car kinematic
function pos_next = kinematic_model(v,delta,pos_current,L,Ts)
%%
% Inputs:
%         v: velocity
%         delta: steering
%         pos_current: current position [x;y;Phi]
%          L: length of wheelbase (m)
%          Ts: sampling time
theta = pos_current(3);

x_dot = v * cos(delta + theta);
y_dot = v * sin(delta + theta);
theta_dot = v * sin(delta) / L;

pos_dot = [x_dot,y_dot,theta_dot].';
pos_next = pos_current + pos_dot*Ts;