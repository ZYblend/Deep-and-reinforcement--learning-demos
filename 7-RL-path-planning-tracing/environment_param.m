%% globle parameters
L = 0.256;     % length of wheelbase (m)
W = 0.5;       % vehicle width
H = 0.5;       % vehicle length

sat_steering = pi/6;

Ts = 0.01;  % control frequency 1/Ts Hz
tot = 1000;

% Lane
Trajectory_generator;                   
x0d = 0;
x1_star = [x0d; g1(x0d); atan(g_dot1(x0d))];  % initilization of trajectory
x2_star = [x0d; g2(x0d); atan(g_dot2(x0d))];
x_target = lane1_waypoints(1,end);
y_target = lane1_waypoints(2,end);

% initial conditions
x0 = x1_star;              % [x, y, heading]  
break_flag = 0;

%% simulation parameter of ego vehicle
iter = 1;

lane_idx = 1;

pos_current = x0;
pos_next = x0;
xi = 0;

delta = 0;
v = 2;
v_min = 1;
v_max = 3;

X_cache = zeros(3,tot);
u_cache = zeros(1,tot);

% controller
k = 1;
ks = 1e-4;
e1 = 0;
e2 = 0;

%% surrounding cars
iter1 = 0;
pos_current1 = [lane1_waypoints(1,round(tot/3)); lane1_waypoints(2,round(tot/3)); pi/4];
pos_next1 = pos_current1;

delta1 = 0;
v1 = 1;

X_cache1 = zeros(3,tot);

%% reward
Reward_cache = zeros(tot,1);
action = 1;