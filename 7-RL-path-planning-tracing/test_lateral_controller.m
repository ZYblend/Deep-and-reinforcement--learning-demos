%% load parameters
clear all
clc

environment_param;
DQN_param;


for ii = 1:episode
    % Initialized state for each episode
    Input_state = Initial_state;
    % initialize all environment paraemeterss
    environment_param;
    disp(ii);
    
    while iter <= tot
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% DQN action
        if rand() < epsilon
           action_value = randi(action_num);
        else
            act_values = predict(dqn_net, Input_state.');
            action_value = find(act_values == max(act_values));   % obtain the action index corresponding to max q value
            action_value = action_value(1);
        end

        lane_idx = Action(action_value);
        u_cache(iter) = lane_idx;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% host car
        % stanley lateral controller
        delta = -e2 + atan(k*e1/(ks+v));

        % steering saturation
        if abs(delta) > sat_steering
            delta =  sign(delta) * sat_steering;
        end
        % velocity saturation
        v = v*(v>=v_min && v<=v_max) + v_min*(v<v_min) + v_max*(v>v_max);

        % kinmeatic model (centered at fron axis)
        pos_next = kinematic_model(v,delta,pos_current,L,Ts);
        X_cache(:,iter) = pos_next;

        % crosstrack error and yaw error
        if lane_idx == 1
            [e1,e2,~,~] = crosstrack_error(pos_next(1),pos_next(2),p1,pos_next(3));
        else
            [e1,e2,~,~] = crosstrack_error(pos_next(1),pos_next(2),p2,pos_next(3));
        end

        % update cache
        pos_current = pos_next;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% surrounding cars
        % steering saturation
        if abs(delta1) > sat_steering
            delta1 =  sign(delta1) * sat_steering;
        end
        
        % kinematics
        pos_next1 = kinematic_model(v1,delta1,pos_current1,L,Ts);

        % car 1 shoul stop before arriving the end point, otherwise, ego
        % car will always crash with car 1 at end point
        if pos_next1(1) >= lane1_waypoints(1,end-round(tot/2))
            pos_next1(1) = lane1_waypoints(1,end-round(tot/2));
            pos_next1(2) = lane1_waypoints(2,end-round(tot/2));
        end

        X_cache1(:,iter) = pos_next1;
        pos_current1 = pos_next1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Rewards    
        % taregt distance
        if ( g1(pos_next(1))+0.5-pos_next(2) ) * ( g1(x_target)+0.5-y_target ) >0   % lane preference, if the car is on the same lane as the target waypoint, give +5 reward
            R1 = 5;
        else                        % if the car is not on the same lane as the target waypoint, give -5 reward
            R1 = -5;
        end
        
        % collision avoidance
        distance1 = sqrt((pos_current(1)-pos_current1(1))^2+(pos_current(2)-pos_current1(2))^2);   % distance from host car to ssurrounding car
        if pos_next(1)<= pos_next1(1)                                                              % if host car is behind surrounding car
            if ( g1(pos_next(1))+0.5-pos_next(2) ) * ( g1(pos_next1(1))+0.5-pos_next1(2) ) < 0     % if host car is not at the same lane as surrounding car, give 0 reward
                R2 = 0;
            else                                                                                   % if host car is at the same lane as surrounding car
                if distance1 <= 10                                                                 % if the ditance is less then detection range (10),the collison probability is started to be calculated
                    R2 = -10*exp(-abs(distance1-max(W,H))^2);                                      % - 10* collision probability is the reward for collision

                end
                if distance1 <= 10*max(W,H)                                                        % if distance is less than warning distance (10倍 car length), give -15 reward
                        R2 = -15;                       
                end
            end
        else
            R2 = 0;                                                                                % if host car is ahead surrounding car, reward is 0
        end
        if distance1 <= max(W,H)           % crash
                disp('collision with surrounding car at time step');                              % if reward is less than car length, car cashed, give reward -50
                disp(iter);
                R2 = -50;
                break_flag = 1;
        end
            
        % outside lane
        if g1(pos_next(1))-0.5 >= pos_next(2) || g2(pos_next(1))+0.5 <= pos_next(2)            % if outsid lane, give reward -20
            disp('outside the lane at time step');
            disp(iter);
            R3 = -20;
            break_flag = 1;
        else
            R3 = 0;
        end
        % outside map
        if pos_next(1)<0 || pos_next(1)>end_x || pos_next(2)<-1 || pos_next(2)>end_x+1        % if car drive out the map and the out point is not the target waypooint, give reward -20
            disp('outside the map at time step');
            disp(iter);
            R4 = -20;
            break_flag = 1;
        else
            R4 = 0;
        end
        
        
        % success
        if abs(pos_next(1)-x_target)<0.1 && abs(pos_next(2) - y_target)<0.05    % successfully reach the target, give erward +100
            disp('Success, End!')
            cvg_flag = cvg_flag + 1;
            R0 = 300;
            break_flag = 1;
            success_path = [success_path;X_cache];
            success_epoch = [success_epoch,epoch_cnt];
            other_car_path = [other_car_path;X_cache1];
            success_action = [success_action;u_cache];
%             save success.mat X_cache
%             save other_car.mat X_cache1
%             save action.mat u_cache
%             save success_epoch.mat epoch_cnt
            for iii = 1:size(X_cache,2)
                figure(1);
                [x_plot,y_plot] = draw_car(X_cache(1,iii),X_cache(2,iii),X_cache(3,iii),W,H);
                hold on, plot(x_plot,y_plot,'r')
                [x_plot,y_plot] = draw_car(X_cache1(1,iii),X_cache1(2,iii),X_cache1(3,iii),W,H);
                hold on, plot(x_plot,y_plot,'k')
            end
        else
            R0 = 0;
        end  
        
            
        Reward = R0 + R1 + R2 + R3 + R4;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Training
        Next_state = [pos_next; x_target; y_target; pos_next1; v1];

        % Put the traj in the memory
        index = rem(memory_count, memory_size) + 1;
        memory{index} = {Input_state.', action_value, Reward, Next_state.'};
        memory_count = memory_count +1;
        
        % If memory full, train dqn with trajs in memory
        if memory_count > memory_size && rem(memory_count,batch_size) == 0 && episode~=1
            
            batch_current_state = zeros(batch_size,state_num);
            batch_target = zeros(batch_size, action_num);
            random_chose_memory = cell(batch_size, 1);
            for i = 1:batch_size
                random_memory_index = randperm(memory_size, 1);
                random_chose_memory{i} = memory{random_memory_index};
                current_state = random_chose_memory{i}{1};
                batch_current_state(i,:) = current_state;
                action_v = random_chose_memory{i}{2};
                reward = random_chose_memory{i}{3};
                next_state = random_chose_memory{i}{4};
                target_f = predict(dqn_net, current_state);
                target = predict(target_net, next_state);
                target_t = reward + gamma .* max(target); 
                target_f(1,action_v) = target_t;
                batch_target(i,:) = target_f;
            end
            [dqn_net, info] = trainNetwork(batch_current_state, batch_target, dqn_net.Layers, options);
            RMSE_log = [RMSE_log info.TrainingRMSE(30)];
            Loss_log = [Loss_log info.TrainingLoss(30)];
            fprintf('epoch: %d',epoch_cnt);
            fprintf('    RMSE %f',info.TrainingRMSE(30));
            fprintf('    Loss %f\n',info.TrainingLoss(30));
            epoch_cnt = epoch_cnt+1;
            epsilon = max([epsilon_min, epsilon * epsilon_decay]);
        end

        if rem(memory_count, update_freq) == 0     % update target dqn
           target_net = dqn_net;
        end

        Input_state = Next_state;
        
        if break_flag == 1
            break;
        end
      
        
        %% plotting
%         figure(1);
%         [x_plot,y_plot] = draw_car(pos_next(1),pos_next(2),pos_next(3),W,H);
%         hold on, plot(x_plot,y_plot,'r')
%         [x_plot,y_plot] = draw_car(pos_next1(1),pos_next1(2),pos_next1(3),W,H);
%         hold on, plot(x_plot,y_plot,'k')
        if ii == episode
            figure(1);
            [x_plot,y_plot] = draw_car(pos_next(1),pos_next(2),pos_next(3),W,H);
            hold on, plot(x_plot,y_plot,'r')
            [x_plot,y_plot] = draw_car(pos_next1(1),pos_next1(2),pos_next1(3),W,H);
            hold on, plot(x_plot,y_plot,'k')
        end
    iter = iter +1;
    end
    if cvg_flag >= 10
        disp('DQN converge at episode');
        disp(ii);
        save success_path.mat success_path;
        save success_epoch.mat success_epoch;
        save other_car_path.mat other_car_path;
        save success_action.mat success_action
        break;
    end
    Iter_cache(ii,1) = iter;
    
    
end
%% plotting
% hold on, plot(X_cache(1,:),X_cache(2,:));
epoch_axis = [1:1:epoch_cnt];
figure(2);
xlabel('Epoch') 
ylabel('Training RMSE') 
plot(epoch_axis,RMSE_log,'LineWidth',2)
figure(3);
xlabel('Epoch') 
ylabel('Training loss') 
plot(epoch_axis,Loss_log,'LineWidth',2)
%% save the dqn model
save dqn_net



