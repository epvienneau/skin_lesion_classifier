file = 'loss_outputs/loss_2018-10-07_01:15:10';
training = xlsread(file, 'A540:A549');
testing = xlsread(file, 'A551:A560');
%%
%figure(1), clf
%plot(num(1:21:end)), hold on, plot(num2), hold off
figure(2), clf
plot(training(1:21:end), '-*', 'LineWidth', 1.5)
grid on
xlabel('epochs')
ylabel('Cross Entropy Loss')
title('Training Loss')
figure(3), clf
plot(testing, '-*', 'LineWidth', 1.5)
grid on
xlabel('epochs')
ylabel('Cross Entropy Loss')
title('Testing Loss')
