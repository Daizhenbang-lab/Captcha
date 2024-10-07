tic
data = importdata('/Users/me/Documents/MATLAB/mini_project/Train/labels.txt');
img_nrs = data(:,1);
true_labels = data(:,(2:4));

my_labels = zeros(size(true_labels));
N = size(img_nrs);

for n = 1:N
    k = img_nrs(n);
    im = imread(sprintf('/Users/me/Documents/MATLAB/mini_project/Train/captcha_%04d.png', k));
    my_labels(k,:) = myclassifier(im);
end

fprintf('\n\nAccuracy: \n');
fprintf('%f\n\n',mean(sum(abs(true_labels - my_labels),2)==0));
toc

