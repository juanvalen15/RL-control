%% System model dynamics

% Description: 
% - medical eye surgery pump model used in RL control
% - position model 

% F = d/dt(V)
% F = A(u)*V + B*u + Bd*d
% F = [
%      Fincomp
%      Finpump
%      Finasptb
%      Finirrtb
%      Fineye
%     ]
