function layers = dqn_model(state_num,action_num)
   %% 创建神经网络
    %包括两个卷积层,两个全连接层
    %图层数组
    layers = [     
        featureInputLayer(state_num,'Normalization','none','Name','input') % 建立输入层

        fullyConnectedLayer(32, 'Name', 'fc1') %全连接层 
        reluLayer('Name','relu1')  %激活函数

        fullyConnectedLayer(32, 'Name', 'fc2') %全连接层
        reluLayer('Name','relu2')  %激活函数

        fullyConnectedLayer(action_num, 'Name', 'output fc') %输出每个动作的Q值
        % 总共有5*30*2 = 120个动作
%         softmaxLayer
        regressionLayer('Name', 'output') %回归层
%          classificationLayer('Name', 'classification')
        ];

end