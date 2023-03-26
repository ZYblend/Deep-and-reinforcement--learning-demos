%% plotting

environment_param;

load action.mat
load other_car.mat
load success.mat

%% image
for iii = 1:499
    figure(1);
    [x_plot,y_plot] = draw_car(X_cache(1,iii),X_cache(2,iii),X_cache(3,iii),W,H);
    hold on, plot(x_plot,y_plot,'r')
    if iii ~= 1
        [x_plot,y_plot] = draw_car(X_cache1(1,iii),X_cache1(2,iii),X_cache1(3,iii),W,H);
        hold on, plot(x_plot,y_plot,'k')
    end
end


%% specific location
% start
figure (2)
subplot(2,3,1)
environment_param;

load action.mat
load other_car.mat
load success.mat

x1 = X_cache(:,1);
x2 = X_cache1(:,1);

[x_plot,y_plot] = draw_car(x1(1),x1(2),x1(3),W,H);
plot(x_plot,y_plot,'r')
[x_plot,y_plot] = draw_car(x2(1),x2(2),x2(3),W,H);
hold on, plot(x_plot,y_plot,'k')
title('start')


% second
subplot(2,3,2)
environment_param;

load action.mat
load other_car.mat
load success.mat

x1 = X_cache(:,148);
x2 = X_cache1(:,148);

[x_plot,y_plot] = draw_car(x1(1),x1(2),x1(3),W,H);
plot(x_plot,y_plot,'r')
[x_plot,y_plot] = draw_car(x2(1),x2(2),x2(3),W,H);
hold on, plot(x_plot,y_plot,'k')
title('step 148')

% Third
subplot(2,3,3)
environment_param;

load action.mat
load other_car.mat
load success.mat

x1 = X_cache(:,220);
x2 = X_cache1(:,220);

[x_plot,y_plot] = draw_car(x1(1),x1(2),x1(3),W,H);
plot(x_plot,y_plot,'r')
[x_plot,y_plot] = draw_car(x2(1),x2(2),x2(3),W,H);
hold on, plot(x_plot,y_plot,'k')

title('step 220')

% forth
subplot(2,3,4)
environment_param;

load action.mat
load other_car.mat
load success.mat

x1 = X_cache(:,301);
x2 = X_cache1(:,301);

[x_plot,y_plot] = draw_car(x1(1),x1(2),x1(3),W,H);
plot(x_plot,y_plot,'r')
[x_plot,y_plot] = draw_car(x2(1),x2(2),x2(3),W,H);
hold on, plot(x_plot,y_plot,'k')
title('step 301')


% fifth
subplot(2,3,5)
environment_param;

load action.mat
load other_car.mat
load success.mat

x1 = X_cache(:,361);
x2 = X_cache1(:,361);

[x_plot,y_plot] = draw_car(x1(1),x1(2),x1(3),W,H);
hold on, plot(x_plot,y_plot,'r')
[x_plot,y_plot] = draw_car(x2(1),x2(2),x2(3),W,H);
hold on, plot(x_plot,y_plot,'k')
title('step 361')

% last
subplot(2,3,6)
environment_param;

load action.mat
load other_car.mat
load success.mat

x1 = X_cache(:,499);
x2 = X_cache1(:,499);

[x_plot,y_plot] = draw_car(x1(1),x1(2),x1(3),W,H);
hold on, plot(x_plot,y_plot,'r')
[x_plot,y_plot] = draw_car(x2(1),x2(2),x2(3),W,H);
hold on, plot(x_plot,y_plot,'k')
title('step 499')


%% plot loss
load dqn_net.mat
figure (6)
subplot(1,2,1)
plot(RMSE_log)
title('training RMSE error')
subplot(1,2,2)
plot(Loss_log)
title('training loss')