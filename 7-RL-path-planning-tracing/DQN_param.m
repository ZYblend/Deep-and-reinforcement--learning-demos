%% DQN
cvg_flag = 0; 

batch_size = 16; 
update_freq = 200; 
gamma = 0.9; 
tao = 0.01;        
rho = 0.3;
epsilon =0.5;            % more exploration initially
epsilon_min = 0.1;  
epsilon_decay = 0.99; 

% action space
Action = [1,2];
num_act = length(Action);

action_num = num_act;

% state space
Initial_state = [pos_current; x_target; y_target; pos_next1; v1];
state_num = length(Initial_state); 
Initial_action_value = zeros(1,action_num);

% define network
dqn_net = dqn_model(state_num,action_num);  % dqn_model.m load dqn network structure
options = trainingOptions('adam', ...
                'MaxEpoch',30, ...  % retrain same batch 30 times
                'MiniBatchSize',batch_size, ...
                'verbose', 0); 

Reward = 0;
dqn_net = trainNetwork(Initial_state.', Initial_action_value, dqn_net, options);
target_net = dqn_net;

memory_size = 4096;  
memory_count = 0;
memory = cell(1,memory_size);  

episode = 500;
Iter_cache = zeros(episode,1);
RMSE_log = [];
Loss_log = [];
success_path = [];
success_epoch = [];
other_car_path = [];
success_action = [];
epoch_cnt = 0;

% load dqn_net.mat

% cvg_flag = 0;