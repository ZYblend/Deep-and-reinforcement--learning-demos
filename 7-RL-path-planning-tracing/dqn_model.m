function layers = dqn_model(state_num,action_num)
   %% ����������
    %�������������,����ȫ���Ӳ�
    %ͼ������
    layers = [     
        featureInputLayer(state_num,'Normalization','none','Name','input') % ���������

        fullyConnectedLayer(32, 'Name', 'fc1') %ȫ���Ӳ� 
        reluLayer('Name','relu1')  %�����

        fullyConnectedLayer(32, 'Name', 'fc2') %ȫ���Ӳ�
        reluLayer('Name','relu2')  %�����

        fullyConnectedLayer(action_num, 'Name', 'output fc') %���ÿ��������Qֵ
        % �ܹ���5*30*2 = 120������
%         softmaxLayer
        regressionLayer('Name', 'output') %�ع��
%          classificationLayer('Name', 'classification')
        ];

end