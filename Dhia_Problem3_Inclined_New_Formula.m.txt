function Dhia_Problem_Inclined_NewFormula

format long g 

%Define all parameters 
global phi1 phi2 Pr lambda sigma1 sigma2 L Bv BT m Eckert s epsilon A1 A2 A3 A4 A5 A6 A7 A8 Gr X %B_1 B_2 B_f

%Boundary layer thickness & stepsize
etaMin = 0;
etaMax1 = 6; %blt 1st solution
etaMax2 = 9; %blt 2nd solution
stepsize1 = etaMax1;
stepsize2= etaMax2;

%Input for the parameters

phi1 = 0.01; %nanoparticle volume fraction
phi2 = 0.01;
Pr = 6.2;  %Prandtl number %Pr=6.2 water
lambda = -1; %stretching/shrinking parameter
L = 1; %mass concentration
Bv = 0.5; %fluid particle interaction (velocity)
BT = 0.5; %fluid particle interaction (temperature)
m = 1.0; %mass concentration fixed to 1
Eckert = 2.0; % Eckert number
epsilon = 1.0; %ratio specific heat
s = 2; %suction
sigma1 = 0.2; %velocity slip parameter
sigma2 = 0.2; %thermal slip parameter
Gr = 0.001; %mixed convection = LAMBDA*
X = 0; %inclination angle in rad

rho_f = 997.1; k_f = 0.6130; C_f = 4179; %base fluid (water)
rho_1 = 3970; k_1 = 40; C_1 = 765;  %Al2O3
rho_2 = 8933; k_2 = 400; C_2 = 385; %Cu
B_1=0.85; % BETA ALUMINA
B_2=1.67;% BETA CU
B_f=21; % BETA FLUID

A1 = 1/[((1-phi1)^(2.5))*((1-phi2)^(2.5))];
A2 = (1-phi2)*[(1-phi1)+phi1*(rho_1/rho_f)]+(phi2*(rho_2/rho_f));
A3 = (1-phi2)*[(1-phi1)+phi1*((rho_1*C_1)/(rho_f*C_f))]+(phi2*((rho_2*C_2)/(rho_f*C_f)));
A4 = (k_1+2*k_f-2*phi1*(k_f-k_1))/(k_1+2*k_f+phi1*(k_f-k_1));
A5 = [(k_2+(2*A4*k_f)-2*phi2*(A4*k_f-k_2))/(k_2+(2*A4*k_f)+phi2*(A4*k_f-k_2))]*A4;
A6=[(1-phi2)*[(1-phi1)*(rho_f*B_f)+phi1*(rho_1*B_1)+phi2*(rho_2*B_2)]]/[(1-phi2)*[(1-phi1)*(rho_f)+phi1*(rho_1)+phi2*(rho_2)]];%BHNF
%A6=[(1-phi1-phi2)*(rho_f*B_f)+phi1*(rho_1*B_1)+phi2*(rho_2*B_2)]/[(1-phi1-phi2)*rho_f+phi1*rho_1+phi2*rho_2];
A7=B_f;
A8=A6/A7;




%%%%%%%%%%%%%%%%%%%%%%   first solution   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    options = bvpset('stats','off','RelTol',1e-10); 
    solinit = bvpinit (linspace (etaMin, etaMax1, stepsize1), @OdeInit1);
    sol = bvp4c (@OdeBVP, @OdeBC, solinit, options);
    eta = linspace (etaMin, etaMax1, stepsize1);
    y = deval (sol, eta);
     
     figure(1)
     plot(sol.x,sol.y(2,:),'r')
     xlabel('\eta')
     ylabel('f`(\eta)')
     hold on

    figure(2)
    plot(sol.x,sol.y(5,:),'b')
    xlabel('\eta')
    ylabel('F`(\eta)')
    hold on
    
    figure(3)
     plot(sol.x,sol.y(6,:),'y')
     xlabel('\eta')
     ylabel('t(\eta)')
     hold on
    
    figure(4)
    plot(sol.x,sol.y(8,:),'k')
    xlabel('\eta')
    ylabel('tp(\eta)')
    hold on

    %Saving the output in txt file for first solution
    descris = [sol.x; sol.y];
    save 'upper_HN.txt' descris -ascii
   
    %Displaying the output for first solution 
    fprintf('\nFirst solution:\n');
    fprintf('A1 = %7.9f\n',A1);
    fprintf('A5 = %7.9f\n',A5);
    fprintf('f"(0) = %7.9f\n',y(3));
    fprintf('F`(0) = %7.9f\n',y(5));
    fprintf('-t`(0) = %7.9f\n',-y(7));
    fprintf('tp(0) = %7.9f\n',y(8));
    fprintf('\n');

%%%%%%%%%%%%%%%%%%%%%%   second solution   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    options = bvpset('stats','off','RelTol',1e-10); 
    solinit = bvpinit (linspace (etaMin, etaMax2, stepsize2), @OdeInit2);
    sol = bvp4c (@OdeBVP, @OdeBC, solinit, options);
    eta = linspace (etaMin, etaMax2, stepsize2);
    y = deval (sol, eta);
     
    figure(1)
    plot(sol.x,sol.y(2,:),'--r')
    xlabel('\eta')
    ylabel('f`(\eta)')
    hold on
    
    figure(2)
    plot(sol.x,sol.y(5,:),'--b')
    xlabel('\eta')
    ylabel('F`(\eta)')
    hold on
    
    figure(3)
    plot(sol.x,sol.y(6,:),'--y')
    xlabel('\eta')
    ylabel('t(\eta)')
    hold on
    
    figure(4)
    plot(sol.x,sol.y(8,:),'--b')
    xlabel('\eta')
    ylabel('tp(\eta)')
    hold on
   
    %Saving the output in txt file for second solution
    descris = [sol.x; sol.y];
    save 'lower_HN.txt' descris -ascii
     
    %Displaying the output for second solution 
    fprintf('\nSecond solution:\n');
    fprintf('f"(0) = %7.9f\n',y(3));
    fprintf('F`(0) = %7.9f\n',y(5));
    fprintf('-t`(0)= %7.9f\n',-y(7));
    fprintf('tp(0) = %7.9f\n',y(8));
    fprintf('\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%Define the ODE function
function ff = OdeBVP (x, y, Pr, L, Bv, BT, m, Eckert, epsilon, A1, A2, A3, A5,A8, Gr, X)

global Pr L Bv BT m Eckert epsilon A1 A2 A3 A5 A8 Gr X

 ff = [y(2)   
       y(3)
      (A2/A1)*(-y(1)*y(3)+((y(2))^2)-((L*Bv)/A2)*(y(5)-y(2))-(Gr*A8*y(6))*(cos(X)))
       y(5)
       (1/y(4))*((y(5))^2-Bv*(y(2)-y(5)))
       y(7)
       ((Pr*A3)/A5)*(-y(1)*y(7)+2*y(2)*y(6)-((L*BT)/(m*A3))*(y(8)-y(6))-((L*Bv*Eckert)/(m*A3))*((y(5)-y(2))^2)-((A1/A3)*(Eckert*(y(3)^2))))
       (1/y(4))*(2*y(5)*y(8)-epsilon*BT*(y(6)-y(8)))];


%Define the boundary condition
function res = OdeBC (ya, yb, lambda, s, sigma1, sigma2)

global lambda s sigma1 sigma2
res = [ ya(1)-s
        ya(2)-lambda-sigma1*ya(3)
        ya(6)-1-sigma2*ya(7)
        yb(2)
        yb(5)
        yb(4)-yb(1)
        yb(6)
        yb(8)];           
 
%Setting the initial guess for first solution  
function v = OdeInit1 (x, lambda)

global lambda 
%1
%     v = [1.2   %0.98 0.2 1.8 1.2
%          0
%          0
%          0
%          0
%          0
%          0
%          0];   %3.21
     
%2
% v = [-2.45*exp(-x) % lower branch 0.0112 1.112
%       exp(-x)   %-
%       -exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       -exp(-x)
%       exp(-x)];  

%3
% v = [-1.1167/exp(-x)-exp(-x)  
%        exp(-x)
%        exp(-x)
%        0.15+exp(-x)
%        exp(-x)
%        exp(-x)
%        exp(-x)
%        0.15+exp(-x)]; 

%4
% v = [-1.1167/exp(-x)-exp(-x)  
%         exp(-x)
%         exp(-x)
%         0.15+exp(-x)
%         exp(-x)
%         exp(-x)
%         exp(-x)
%         0.15+exp(-x)]; 

%5 
%v = [1.1124+exp(-x) % lower branch 0.0112 1.112
%       exp(-x)   %-
%      exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)];  

% %6
v = [1.112+exp(-x)*exp(-x) 
        exp(-x)
        exp(-x)
        0.15+exp(-x)
        exp(-x)
        exp(-x)
        exp(-x)
        0.15+exp(-x)]; 

%7
% v = [-sin(pi*x)
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)];  

%8
% v = [exp(-x)*exp(-x)+1 
%         0.03*exp(-x)
%         1-exp(-x)
%         0.15+exp(-x)
%         exp(-x)
%         exp(-x)
%         exp(-x)
%         0.15+exp(-x)]; 


%Setting the initial guess for second solution 
function v1 = OdeInit2 (x, lambda, s)

global lambda s

%1 
%v1 = [7.4273/exp(-x) % lower branch 1.12 1.2 0.912 0.7912
%       exp(-x)   %-
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)];  

%2    
%v1 = [1.12+exp(-x) %contoh coding senang dapat
%         exp(-x)
%         exp(-x)
%         exp(-x)
%         exp(-x)
%         0.5+exp(-x)
%         exp(-x)
%         0.5+exp(-x)]; 

%3
%v1 = [3.5/exp(-x)  %lambda 3-2.6 (1.0)
%        exp(-x)
%        exp(-x)
%        exp(-x)
%        exp(-x)
%        1.5+exp(-x)
%        exp(-x)
%        1.5+exp(-x)]; 

%4
%v1  = [1.63  % run M=0.1
%         0  % 
%         0
%         0
%         0
%         0
%         0
%         0];

%6    
%v1 = [1.13+exp(-x) %contoh coding senang dapat
%         exp(-x)
%        exp(-x)
%         exp(-x)
%         exp(-x)
%         0.1+exp(-x)
%         exp(-x)
%         0.1+exp(-x)]; 
 
% %7 
v1 = [3.912/exp(-x) % lower branch 1.12 1.2 0.912 0.7912
       exp(-x)   %-
       exp(-x)
       exp(-x)
       exp(-x)
       exp(-x)
       exp(-x)
       exp(-x)];  

%8
%v1 = [1-exp(-x) % lower branch Alumina and titania
%       1+exp(-x)
%       1+exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)];

%9
%v1 = [-1+exp(-x)
%       exp(-x)   
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)
%       exp(-x)];  

%10
% v1 = [1-exp(-x)
%        exp(-x)
%        exp(-x)
%        exp(-x)
%        exp(-x)
%        exp(-x)
%        exp(-x)
%        exp(-x)];  

