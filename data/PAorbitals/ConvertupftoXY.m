clc;
clear;
Xdata = load("Na_X.txt");
Ydata = load("Na_3s.txt");

flag = 1;
for i = 1:length(Xdata(:,1))
    for j = 1:length(Xdata(1,:))
        
        if(Xdata(i,j) == -999999)
            flag = 0;
            break;
        end
        X((i-1)*length(Xdata(1,:))+j,1) = Xdata(i,j);
    end
    if(flag == 0)
        break;
    end
end
flag = 1;
for i = 1:length(Ydata(:,1))
    for j = 1:length(Ydata(1,:))
        
        if(Ydata(i,j) == -999999)
            flag = 0;
            break;
        end
        Y((i-1)*length(Ydata(1,:))+j,1) = Ydata(i,j);
    end
    if(flag == 0)
        break;
    end
end

Z = [X,Y];
writematrix(Z, "PA_11_3_0.txt");


