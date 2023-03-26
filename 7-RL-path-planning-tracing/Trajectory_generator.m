%% Generate some interesting trajactory


%% single second-order polynominal trajectory
double_lane_flag = 0;
f = @(x) x;            % striaght line
% f = @(x) 0.01*x.^2;      % quandratic curve
% f = @(x) 0.3*sin(0.1*x);   % sin curve
end_x = 7;
z = linspace(0,end_x,tot);

p1 = polyfit(z,f(z),2);
g1 = @(x) p1(1)*x.^2 + p1(2)*x + p1(3)- 0.5;
g2 = @(x) p1(1)*x.^2 + p1(2)*x + p1(3)+ 0.5;
p2 = [p1(1),p1(2),p1(3)+ 0.5];
p1 = [p1(1),p1(2),p1(3)- 0.5];
g_dot1 = @(x) 2*p1(1)*x + p1(2);
g_dot2 = @(x) 2*p1(1)*x + p1(2);

lane1_waypoints = [z;g1(z);g_dot1(z)];
lane2_waypoints = [z;g2(z);g_dot2(z)];

% plot(z,f(z),'o');
plot(z,g1(z),'r--')
hold on, plot(z,g1(z)-0.5,'k-')
hold on, plot(z,g1(z)+0.5,'b-')
hold on, plot(z,g2(z),'r--')
hold on, plot(z,g2(z)+0.5,'k-')
grid on



% 
% %% Double Second-order polynominal trajectory
% double_lane_flag = 1;
% 
% f = @(x) 3*sin(0.1*x);   % sin curve
% 
% x_boundary = 30;
% z1 = linspace(0,x_boundary,1000);
% z2 = linspace(x_boundary,2*x_boundary,1000);
% 
% p1 = polyfit(z1,f(z1),2);
% p2 = polyfit(z2,f(z2),2);
% 
% g1 = @(x) p1(1)*x.^2 + p1(2)*x + p1(3);
% g2 = @(x) p2(1)*x.^2 + p2(2)*x + p2(3);
% 
% g_dot1 = @(x) 2*p1(1)*x + p1(2);
% g_dot2 = @(x) 2*p2(1)*x + p2(2);
% 
% %plot(z1,f(z1),'o');
% %hold on, plot(z2,f(z2),'o');
% hold on, plot(z1,g1(z1),'LineWidth',2);
% hold on, plot(z2,g2(z2),'LineWidth',2);