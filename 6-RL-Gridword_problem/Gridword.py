import matplotlib.pyplot as plt
import numpy as np



class Gridworld:
    def __init__(self):
        self.row_size = 5
        self.column_size = 5
        self.states = 25

        self.actions = ['N','S','E','W']
        self.policy = np.ones((self.row_size,self.column_size,4),dtype=int)   
                    # 3d array, the first 2 axis represents the grid, the last 1 axis has 4 dimension (0 means no action, 1 means have the action)
                    #                                                                                  N,S,E,W
                    # Initialized as random policy

        self.values = np.zeros((self.row_size,self.column_size))
        self.gamma = 0.9

        self.reward_off_grid = -1
        self.reward_AA_prime = 5
        self.reward_BB_prime = 10
        self.reward_nomral = 0

        self.covergency_tol = 0.001

    def environment(self,row,col,act):
        ## move north
        if act == self.actions[0]:
            if row == 0: 
                state_update = [row,col]
                reward = self.reward_off_grid
            else:
                state_update = [row-1,col]
                reward = self.reward_nomral
        ## move south
        elif act == self.actions[1]:
            if row == self.row_size-1: 
                state_update = [row,col]
                reward = self.reward_off_grid
            else:
                state_update = [row+1,col]
                reward = self.reward_nomral
        ## move east
        elif act == self.actions[2]:
            if col == self.column_size-1: 
                state_update = [row,col]
                reward = self.reward_off_grid
            else:
                state_update = [row,col+1]
                reward = self.reward_nomral
        ## move west
        elif act == self.actions[3]:
            if col == 0: 
                state_update = [row,col]
                reward = self.reward_off_grid
            else:
                state_update = [row,col-1]
                reward = self.reward_nomral
        ## A-A_prime
        if row == 0 and col == 1:
            state_update = [4,1]
            reward = self.reward_AA_prime
        ## B-B_prime
        if row == 0 and col == 3:
            state_update = [2,3]
            reward = self.reward_BB_prime
        return state_update, reward

    # Given any policy, obtain the estimated value function
    def policy_evaluation(self,values,policy):
        iter = 0 
        gridValue_diff = 1
        while gridValue_diff > self.covergency_tol:
            values_update = np.zeros((self.row_size,self.column_size))
            for row in range(self.row_size):
                for col in range(self.column_size):
                    num_actions = np.sum(policy[row,col,:])
                    policy_act = []
                    for iter in range(4):
                        if policy[row,col,iter] == 1:
                            policy_act.append(self.actions[iter])
                    for act in policy_act:
                        new_state, reward = self.environment(row,col,act)
                        values_update[row,col] += (1/num_actions) * (reward + self.gamma * values[new_state[0],new_state[1]])
            gridValue_diff = np.sum(np.abs(values_update-values))
            values = values_update.copy()
            iter += 1
        print('total iterations for value evaluation: ',iter)
        return values_update


    ## find optimal policy and the corresponding value function through policy iteration
    def policy_iteration(self,values,policy):
        policy_update = policy.copy()
        stable = False
        iter = 0
        while stable == False:
            policy_stable = np.zeros((self.row_size,self.column_size))
            for row in range(self.row_size):
                for col in range(self.column_size):
                    act_values = []
                    for act in self.actions:
                        new_state, reward = self.environment(row,col,act)
                        act_values.append((reward + self.gamma * values[new_state[0],new_state[1]]))
                    badact_indice = np.flatnonzero(act_values!=np.max(act_values))
                    policy_update[row,col,badact_indice] = 0
                    if (policy_update[row,col,:] == policy[row,col,:]).all():
                        policy_stable[row,col] = 1
            values_update = self.policy_evaluation(values,policy_update)
            policy = policy_update.copy()
            values = values_update.copy()
            policy_update = np.ones((self.row_size,self.column_size,4),dtype=int)
            if np.sum(policy_stable) == self.states:
                stable = True
            iter += 1
        print('total iterations for policy iteration: ',iter)
        return policy, values
    

    ## find optimal policy and the corresponding value function through value iteration
    def value_iteration(self,values):
        iter = 0 
        gridValue_diff = 1
        while gridValue_diff > self.covergency_tol:
            values_update = np.zeros((self.row_size,self.column_size))
            for row in range(self.row_size):
                for col in range(self.column_size):
                    action_value = []
                    for act in self.actions:
                        new_state, reward = self.environment(row,col,act)
                        action_value.append(reward + self.gamma * values[new_state[0],new_state[1]])
                    values_update[row,col] = action_value[np.random.choice(np.flatnonzero(action_value==np.max(action_value)))]
            gridValue_diff = np.sum(np.abs(values_update-values))
            values = values_update.copy()
            self.plotting_values(values)
            if iter%10 == 0:
                name = "value_iteration"+str(iter)
                plt.savefig(name)
            plt.close()
            iter += 1
        print('total iterations for value iteration: ',iter)
        # output a deterministic policy
        policy = np.ones((self.row_size,self.column_size,4),dtype=int) 
        for row in range(self.row_size):
            for col in range(self.column_size):
                act_values = []
                for act in self.actions:
                    new_state, reward = self.environment(row,col,act)
                    act_values.append((reward + self.gamma * values[new_state[0],new_state[1]]))
                badact_indice = np.flatnonzero(act_values!=np.max(act_values))
                policy[row,col,badact_indice] = 0
        
        return values, policy


    def plotting_values(self,values):
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

        plt.grid(True)
        plt.axis([0, 5, 0, 5])

        for row in range(self.row_size):
            for col in range(self.column_size):
                value=str(np.round(values[row,col],1))
                plt.text(col+0.5,5-row-0.5,value,ha='center',va='center',fontsize='xx-large')


    def plotting_policy(self,policy):
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

        plt.grid(True)
        plt.axis([0, 5, 0, 5])

        for row in range(self.row_size):
            for col in range(self.column_size):
                for iter in range(policy.shape[2]):
                    if policy[row,col,iter] == 1:
                        if iter == 0:
                            plt.arrow(col+0.5,5-row-0.5,0,0.25,width=0.05)
                        elif iter == 1:
                            plt.arrow(col+0.5,5-row-0.5,0,-0.25,width=0.05)
                        elif iter == 2:
                            plt.arrow(col+0.5,5-row-0.5,0.25,0,width=0.05)
                        elif iter == 3:
                            plt.arrow(col+0.5,5-row-0.5,-0.25,0,width=0.05)
        plt.show()
                        



if __name__ == '__main__':
    Grid = Gridworld()

    ## (1) random policy
    values_rand = Grid.policy_evaluation(Grid.values,Grid.policy)
    Grid.plotting_values(values_rand)
    plt.show()

    ## (2) optimal policy iteration
    policy_opt, values_opt = Grid.policy_iteration(values_rand,Grid.policy)
    Grid.plotting_values(values_opt)
    plt.show()
    Grid.plotting_policy(policy_opt)

    ## (3) optimal value iteration
    values_opt2, policy_opt2 = Grid.value_iteration(values_rand)
    Grid.plotting_values(values_opt2)
    plt.show()
    Grid.plotting_policy(policy_opt2)









