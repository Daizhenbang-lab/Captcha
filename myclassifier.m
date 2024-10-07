function S = myclassifier(I)

load('trained_model.mat');

patterns = [];

I = data_preprocessing(I);
features=feature_extraction(I); % Extract features
if size(features) > 0
    for j=1:3
        patterns(end+1,:) = features(j,:,:);
    end
end
S = predict(Mdl,patterns);
end