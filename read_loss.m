file = 'loss_outputs/final-metrics.xlsx';
training = xlsread(file, 'A540:A549');
testing = xlsread(file, 'A551:A560');
confusion_mat = xlsread(file, 1, 'A490:A538');
%%
figure(1), clf
plot(training, '-*', 'LineWidth', 1.5)
grid on
xlabel('epochs')
ylabel('Cross Entropy Loss')
title('Training Loss')
figure(2), clf
plot(testing, '-*', 'LineWidth', 1.5)
grid on
xlabel('epochs')
ylabel('Cross Entropy Loss')
title('Testing Loss')
confusion_mat = reshape(confusion_mat, 7, 7)
