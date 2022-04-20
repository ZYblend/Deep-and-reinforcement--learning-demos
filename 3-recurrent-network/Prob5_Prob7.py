import numpy as np

class My_network(object):
    def __init__(self):
        pass  

    def softmax(self,x):
        '''Compute softmax values for each sets of scores in x.'''
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    def tanh(self,x):
        return np.tanh(x)


    def feedforward_prop_step(self,h_prev,x, b,c,W,U,V):
        ''' one step feedforward prop 
            implement equations (10.8)-(10.11) in book "Deep Learning, Ian Goodfellow"
            Inputs:
                   h_prev: previous states
                   x: current input
        '''

        a = b + np.dot(W, h_prev) + np.dot(U, x)
        h = np.tanh(a)
        o = c + np.dot(V, h)
        y = self.softmax(o)
        return a,h,o, y

    def feedforward_prop(self,xlist,h0,b,c,W,U,V):
        ylist = []
        T = len(xlist)
        h = h0
        for layers in range(T):
            a,h,o, y = self.feedforward_prop_step(h,xlist[layers],b,c,W,U,V)
            ylist.append(y)
        return ylist
    
    def loss(self,y):
        return (y[1,0]-0.5)**2 - np.log(y[0,0])

    def central_difference_method(self,f,a, epsilon = 0.0005):
        return ( f(a+epsilon) - f(a-epsilon) )/(2*epsilon)
    
    def dL_dy3(self,y):
        return np.array([[-1/y[0][0], 2*(y[1][0]-0.5)]])
    def dy_dh(self,y,V):
        return np.dot(np.array([[y[0][0]*(1-y[0][0]), -y[0][0]*y[1][0]], [-y[1][0]*y[0][0], y[1][0]*(1-y[1][0])]]), V)
    def dh_db(self,h):
        return np.array([[1-h[0][0]**2,0],[0, 1-h[1][0]**2]])
    def dh_dh(self,h,W):
        return np.array([[1-h[0][0]**2,0],[0, 1-h[1][0]**2]]) @ W

    def unfolding_method(self,xlist,h0, b,c,W,U,V):
        # feedforward
        a1, h1, o1, y1 = self.feedforward_prop_step(h0,xlist[0],b,c,W,U,V)
        a2, h2, o2, y2 = self.feedforward_prop_step(h1,xlist[1],b,c,W,U,V)
        a3, h3, o3, y3 = self.feedforward_prop_step(h2,xlist[2],b,c,W,U,V)
        L = self.loss(y3)

        # Gradients
        dL_db3 = self.dL_dy3(y3) @ self.dy_dh(y3,V) @ self.dh_db(h3)
        dL_db2 = self.dL_dy3(y3) @ self.dy_dh(y3,V) @ self.dh_dh(h3,W) @ self.dh_db(h2)
        dL_db1 = self.dL_dy3(y3) @ self.dy_dh(y3,V) @ self.dh_dh(h3,W) @ self.dh_dh(h2,W) @ self.dh_db(h1)

        print('dL/dy3', self.dL_dy3(y3))
        print('dy3/dh3', self.dy_dh(y3,V))
        print('dh3/db3', self.dh_db(h3))
        print('dL/db3', dL_db3)
        print('dh3/dh2',self.dh_dh(h3,W))
        print('dh2/db2',self.dh_db(h2))
        print('dL/db2', dL_db2)
        print('dh2/dh1',self.dh_dh(h3,W))
        print('dh1/db1',self.dh_db(h1))
        print('dL/db1', dL_db1)

        return dL_db1, dL_db2, dL_db3
    
    def LSTM(self,h_prev,x,s_prev,Uf,Wf,bf,U,W, b, Ui,Wi, bi, Uo,Wo, bo):
        # forget gate
        f = self.sigmoid( np.dot(Wf, h_prev) + np.dot(Uf, x) + bf)
        # internal state
        g = self.sigmoid( np.dot(Wi, h_prev) + np.dot(Ui, x) + bi )
        s = f * s_prev + g * self.sigmoid(b + np.dot(W, h_prev) + np.dot(U, x) )
        # output gate
        q = self.sigmoid( np.dot(Wo, h_prev) + np.dot(Uo, x) + bo )
        h = self.tanh(s) * q

        return h, s, f, g, q
    
    def feedforward_step_LSTM(self,h_prev,x, b,c,W,U,V, s_prev,Uf,Wf,bf, Ui,Wi, bi, Uo,Wo, bo):
        h, s, f, g, q = self.LSTM(h_prev,x,s_prev,Uf,Wf,bf,U,W, b, Ui,Wi, bi, Uo,Wo, bo)
        o = c + np.dot(V, h)
        y = self.softmax(o)
        return h, o, y, s, f, g, q

    def feedfoward_LSTM(self,xlist,h0,s0,b,c,W,U,V, Uf,Wf,bf, Ui,Wi, bi, Uo,Wo, bo):
        ylist = []
        T = len(xlist)
        h = h0
        s = s0
        for layers in range(T):
            h, o, y, s, f, g, q = self.feedforward_step_LSTM(h,xlist[layers],b,c,W,U,V, s,Uf,Wf,bf, Ui,Wi, bi, Uo,Wo, bo)
            ylist.append(y)
        return ylist
    
    def dh_dbf3(self,h,s_prev,f):
        dh_ds = np.array([[1-h[0][0]**2],[1-h[1][0]**2]])
        ds_df = s_prev
        df_dbf = f * (1-f)
        return dh_ds * ds_df * df_dbf
    def dh_dh_LSTM(self,h,s_prev,f,Wf):
        return np.array([[1-h[0][0]**2],[1-h[1][0]**2]]) * s_prev * Wf @ (f * (1-f))

    def unfolding_LSTM_forget_gate(self,h0,xlist,s0,b,c,W,U,V, Uf,Wf,bf, Ui,Wi, bi, Uo,Wo, bo):
        # feedforward
        h1, o1, y1, s1, f1, g1, q1 = self.feedforward_step_LSTM(h0,xlist[0],b,c,W,U,V,s0,Uf,Wf,bf, Ui,Wi, bi, Uo,Wo, bo)
        h2, o2, y2, s2, f2, g2, q2 = self.feedforward_step_LSTM(h1,xlist[1],b,c,W,U,V,s1,Uf,Wf,bf, Ui,Wi, bi, Uo,Wo, bo)
        h3, o3, y3, s3, f3, g3, q3 = self.feedforward_step_LSTM(h2,xlist[2],b,c,W,U,V,s2,Uf,Wf,bf, Ui,Wi, bi, Uo,Wo, bo)
        L = self.loss(y3)

        # gradients
        dL_dbf3 = self.dL_dy3(y3) @ self.dy_dh(y3,V) * self.dh_dbf3(h3,s2,f3).T
        dL_dbf2 = self.dL_dy3(y3) @ self.dy_dh(y3,V) * self.dh_dh_LSTM(h3,s2,f3,Wf).T * self.dh_dbf3(h2,s1,f2).T
        dL_dbf1 = self.dL_dy3(y3) @ self.dy_dh(y3,V) * self.dh_dh_LSTM(h3,s2,f3,Wf).T * self.dh_dh_LSTM(h2,s1,f2,Wf).T * self.dh_dbf3(h1,s0,f1).T

        return dL_dbf1, dL_dbf2, dL_dbf3



if __name__ == '__main__':
    ################## Problem 5 ##################
    print('################# Problem 5 ##########################')
    b1 = -1
    b2 = 1
    b = np.array([[b1],[b2]])
    c = np.array([[0.5],[-0.5]])
    W = np.array([[1,-1],[0,2]])
    U = np.array([[-1,0],[1,-2]])
    V = np.array([[-2,1],[-1,0]])

    recurrent_net = My_network()

    ## (1)
    h0 = np.zeros((2,1))   # initialization
    x1 = np.array([[1],[0]])
    x2 = np.array([[0.5],[0.25]])
    x3 = np.array([[0],[1]])
    xlist = [x1, x2, x3]

    ylist = recurrent_net.feedforward_prop(xlist,h0,b,c,W,U,V)
    print('all outputs', ylist)

    L = recurrent_net.loss(ylist[-1])
    print('loss', L)

    ## (2) central difference method
    # for b1
    fb1 = lambda b1: recurrent_net.loss( recurrent_net.feedforward_prop(xlist,h0,np.array([[b1],[b2]]),c,W,U,V)[-1] )
    grad_b1 = recurrent_net.central_difference_method(fb1,b1)
    print('gradient of loss with respect to b1: ', grad_b1)

    # for b2
    fb2 = lambda b2: recurrent_net.loss( recurrent_net.feedforward_prop(xlist,h0,np.array([[b1],[b2]]),c,W,U,V)[-1] )
    grad_b2 = recurrent_net.central_difference_method(fb2,b2)
    print('gradient of loss with respect to b2: ', grad_b2)

    ## (3) unfolding method
    dL_db1, dL_db2, dL_db3 = recurrent_net.unfolding_method(xlist,h0, b,c,W,U,V)
    print('gradient with respect to b', dL_db1 + dL_db2 + dL_db3)

    ## (4) gradient descent
    dL_db =  dL_db1 + dL_db2 + dL_db3
    b = b - 0.002*dL_db.T
    print('updated b after one step of gradient descent', b)

    ## (5) calculate loss with updated b
    ylist = recurrent_net.feedforward_prop(xlist,h0,b,c,W,U,V)
    L = recurrent_net.loss(ylist[-1])
    print('updated loss', L)

    ################## Problem 7 ##################
    print('################# Problem 7 ##########################')
    s0 = np.array([[0.1],[0.1]])
    ylist_LSTM = recurrent_net.feedfoward_LSTM(xlist,h0,s0,b,c,W,U,V, U,W,b, U,W,b, U,W,b)
    L_LSTM = recurrent_net.loss(ylist_LSTM[-1])
    print('outputs obtained with LSTM', ylist_LSTM)
    print('Loss obtained with LSTM', L_LSTM)

    dL_dbf1, dL_dbf2, dL_dbf3 = recurrent_net.unfolding_LSTM_forget_gate(h0,xlist,s0,b,c,W,U,V, U,W,b, U,W,b, U,W,b)
    print('gradient with respect to forget gate bias layer 1', dL_dbf1 )
    print('gradient with respect to forget gate bias layer 2', dL_dbf2)
    print('gradient with respect to forget gate bias layer 3', dL_dbf3)
    print('gradient with respect to forget gate bias', dL_dbf1 + dL_dbf2 + dL_dbf3)



       

