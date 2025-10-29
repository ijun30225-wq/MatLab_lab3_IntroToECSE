%% Lab 03 Part B - Machine Learning Lab Matlab Script Template
% Traffic Sign Classification Using Machine Learning

%% Step 1: Data Import
trainingAnnotationsFolder = '3B_training_image_annotations';
testingAnnotationsFolder = '3B_test_image_annotations';
trainingImagesFolder = '3B_training_images';
testingImagesFolder = '3B_test_images';

[trainingFilenames, trainingLabels] = parseAnnotations(trainingAnnotationsFolder);
[testingFilenames, testingLabels] = parseAnnotations(testingAnnotationsFolder);

% Display labels and sample images
sampleIdx = [1 10 15 25 50];
%{
figure(1)
tiledlayout("horizontal")
for i = 1:5
    nexttile
    imshow(trainingFilenames{sampleIdx(i)})
    title(strcat(trainingFilenames{sampleIdx(i)}, ": ", trainingLabels{sampleIdx(i)}))
end
%}
%% Step 2: Feature Extraction for Training Data;
features = extractFeatures(trainingFilenames, trainingImagesFolder);
trainingLabels = transpose(trainingLabels);
[N,C,S] = normalize(features);

%% Step 3: Train K-Nearest Neighbors (KNN) Model
knnModel = fitcknn(N, trainingLabels, 'NumNeighbors', 4, 'ClassNames', {'stop','crosswalk', 'speedlimit'});

%% Step 4: Feature Extraction for Test Data
features2 = extractFeatures(testingFilenames, testingImagesFolder);
N2 = normalize(features2, "center", C, "scale", S);
answers = predict(knnModel, N2);
%% Step 5: Evaluate the Model
cm = confusionmat(testingLabels, answers);
confusionchart(cm, {'stop','crosswalk','speedlimit'});

%% Function for Feature Extraction
function features = extractFeatures(filenames, folder)
    features = [];
    
    for i = 1:length(filenames)
        imgFile = fullfile(folder, filenames{i});
        img_original = imread(imgFile);
        
        % Convert to appropriate formats
        img_double = im2double(img_original);
        img_gray = rgb2gray(img_double);
        
        % Step 1: Binarize the image
        img_bw = imbinarize(img_gray, 'adaptive', 'Sensitivity', 0.3);
        
        % Step 2: Remove small areas and fill holes
        img_bw = bwareaopen(img_bw, 100);
        img_bw = imfill(img_bw, 'holes');
        
        % Step 3: Find the largest connected component
        stats = regionprops(img_bw, 'Area', 'Circularity', 'BoundingBox', 'SubarrayIdx');
        
        if ~isempty(stats)
            % Find the largest region
            [~, idx] = max([stats.Area]);
            
            % Step 4: Extract shape feature - Circularity
            circularity = stats(idx).Circularity;
            
            % Step 5: Extract color features from the bounded region
            if ~isempty(stats(idx).SubarrayIdx)
                subIdx = stats(idx).SubarrayIdx;
                img_sub = img_original(subIdx{1}, subIdx{2}, :);
                img_sub_double = im2double(img_sub);
                
                R = img_sub_double(:,:,1);
                G = img_sub_double(:,:,2);
                B = img_sub_double(:,:,3);
                
                % Calculate mean color values within the bounded region
                meanR = mean(R(:));
                meanG = mean(G(:));
                meanB = mean(B(:));
                
                % Calculate normalized color ratios (important for classification)
                total_color = meanR + meanG + meanB + eps;
                normR = meanR / total_color;
                normG = meanG / total_color;
                normB = meanB / total_color;
                red_dominance = (meanR - meanB) * 2;
                stats = regionprops(img_bw, 'Area', 'Circularity', 'Eccentricity', 'BoundingBox', 'SubarrayIdx');
                % Then add eccentricity to features:
                feature_vector = [circularity, normR, normG, normB, red_dominance, stats(idx).Eccentricity];
                
                % Feature vector: circularity + normalized RGB
            else
                % If no subarray indices, use default values
                feature_vector = [0, 0.33, 0.33, 0.33];
            end
        else
            % If no regions found, use default values
            feature_vector = [0, 0.33, 0.33, 0.33];
        end
        
        features = [features; feature_vector];
    end
end
