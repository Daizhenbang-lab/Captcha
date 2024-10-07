clear all;

folderPath = '/Users/me/Documents/MATLAB/mini_project/Train/';

% Read the file names in the folder
fileNames = dir(fullfile(folderPath, '*.png'));
labels = csvread('/Users/me/Documents/MATLAB/mini_project/Train/labels.txt');

% Randomly shuffle the file names and labels together
% Set a seed
rng(1); 
shuffledIndices = randperm(length(fileNames));
shuffledFileNames = fileNames(shuffledIndices);
shuffledLabels = labels(shuffledIndices,:);

%split data into training and test sets
trainRatio = 0.9;
cv = cvpartition(length(shuffledIndices), 'HoldOut', 1 - trainRatio);

trainingIndices = training(cv);
validationIndices = test(cv);

train_data = shuffledFileNames(trainingIndices);
validation_data = shuffledFileNames(validationIndices);
train_labels = shuffledLabels(trainingIndices,:);
validation_labels = shuffledLabels(validationIndices,:);


validation_pred = train(folderPath, train_data, validation_data, train_labels, validation_labels, 100);

function validation_pred = train(folderPath, train_data, validation_data, train_labels, validation_labels, splits)
    fprintf('Extracting training features...\n');
    t=tic;

    labelsTrainNr = [];
    train_patterns = [];
    validation_pred = zeros(size(validation_labels(:,2:4)));

    %get patterns from trainings images and get correct labels
    for i=1:size(train_data,1)
        I=imread(join([folderPath,train_data(i).name], '')); 
        I = data_preprocessing(I);
        features=feature_extraction(I); 
        if size(features) > 0
            for j=1:3
                
                train_patterns(end+1,:) = features(j,:,:);
                labelsTrainNr(end+1) = train_labels(i,j+1);
            end
        end
    end

    toc(t) 

    fprintf('Building model...\n');
    t=tic;
    
    tr = templateTree('MaxNumSplits',splits);
    Mdl = fitcensemble(train_patterns,labelsTrainNr, 'Learners',tr); 
    toc(t)
    
    fprintf('\nResubstitution error: %5.2f%%\n\n',100*resubLoss(Mdl));

    save('trained_model.mat', 'Mdl');

    fprintf('Validating model...\n');
    t=tic;

    % evaluate all validation images
    for i=1:size(validation_data,1)
        I=imread(join([folderPath,validation_data(i).name], '')); 
        validation_pred(i,:) = myclassifier(I);
    end


    accuracy = mean(sum(abs(validation_pred - validation_labels(:,2:4)),2)==0);
    fprintf('Validation accuracy for images: %5.2f%%\n',accuracy*100);

    accuracy2 = mean(abs(reshape(validation_labels(:,2:4).',1,[]) - reshape(validation_pred.',1,[]))==0);
    fprintf('Validation accuracy for individual digits: %5.2f%%\n',accuracy2*100);
    
    f=figure(2);
    confusionchart(reshape(validation_labels(:,2:4).',1,[]), reshape(validation_pred.',1,[]), 'ColumnSummary','column-normalized', 'RowSummary','row-normalized');
    title(sprintf('Validation accuracy: %5.2f%%\n',accuracy*100));
end
    