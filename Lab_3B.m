%% Lab 03 Part B - Machine Learning for Traffic Sign Recognition
%
% Authors: Nicholas Gladu, Jun Iguchi
% Course:  ECSE 1010 - Introduction to ECSE, RPI
%
% Classifies road signs into three categories using K-Nearest Neighbors (KNN):
%   • stop
%   • speedlimit
%   • crosswalk
%
% Requirements:
%   - MATLAB Image Processing Toolbox
%   - MATLAB Statistics and Machine Learning Toolbox
%   - parseAnnotations.m on the MATLAB path
%   - The following folders on the MATLAB path:
%       3B_training_images/
%       3B_test_images/
%       3B_training_image_annotations/
%       3B_test_image_annotations/

%% -------------------------------------------------------------------------
%% Step 1: Data Import
%% -------------------------------------------------------------------------
trainingAnnotationsFolder = '3B_training_image_annotations';
testingAnnotationsFolder  = '3B_test_image_annotations';
trainingImagesFolder      = '3B_training_images';
testingImagesFolder       = '3B_test_images';

[trainingFilenames, trainingLabels] = parseAnnotations(trainingAnnotationsFolder);
[testingFilenames,  testingLabels]  = parseAnnotations(testingAnnotationsFolder);

% Display a handful of labelled sample images to verify annotations
sampleIdx = [1 10 15 25 50];
figure;
tiledlayout('horizontal');
for i = 1:numel(sampleIdx)
    nexttile;
    imshow(fullfile(trainingImagesFolder, trainingFilenames{sampleIdx(i)}));
    title(sprintf('%s: %s', trainingFilenames{sampleIdx(i)}, trainingLabels{sampleIdx(i)}), ...
          'Interpreter', 'none');
end
sgtitle('Sample Training Images with Class Labels');

%% -------------------------------------------------------------------------
%% Step 2: Feature Extraction — Training Data
%% -------------------------------------------------------------------------
features       = extractFeatures(trainingFilenames, trainingImagesFolder);
trainingLabels = transpose(trainingLabels);

% Z-score normalise; save centre (C) and scale (S) for test normalisation
[N, C, S] = normalize(features);

%% -------------------------------------------------------------------------
%% Step 3: Train KNN Model
%%   NumNeighbors = 4  (tunable)
%% -------------------------------------------------------------------------
knnModel = fitcknn(N, trainingLabels, ...
                   'NumNeighbors', 4, ...
                   'ClassNames',   {'stop', 'crosswalk', 'speedlimit'});

%% -------------------------------------------------------------------------
%% Step 4: Feature Extraction — Test Data
%% -------------------------------------------------------------------------
features2 = extractFeatures(testingFilenames, testingImagesFolder);

% Apply the same normalisation learned from training data
N2 = normalize(features2, 'center', C, 'scale', S);

answers = predict(knnModel, N2);

%% -------------------------------------------------------------------------
%% Step 5: Evaluate the Model
%% -------------------------------------------------------------------------
cm = confusionmat(testingLabels, answers);

figure;
confusionchart(cm, {'stop', 'crosswalk', 'speedlimit'});
title('KNN Confusion Chart — Traffic Sign Classification');

% Overall accuracy
accuracy = sum(diag(cm)) / sum(cm(:)) * 100;
fprintf('\nModel accuracy: %.1f%%\n', accuracy);

%% =========================================================================
%% Local Function: extractFeatures
%%
%%   For each image:
%%     1. Grayscale conversion + adaptive binarization
%%     2. Small-region removal (bwareaopen) + hole filling (imfill)
%%     3. Largest connected component → bounding box (SubarrayIdx)
%%     4. Shape feature  : Circularity, Eccentricity
%%     5. Colour features: mean R/G/B, normalised ratios, red-dominance score
%%
%%   Returns an N-by-6 feature matrix (one row per image).
%% =========================================================================
function features = extractFeatures(filenames, folder)

    features = [];

    for i = 1:length(filenames)
        imgFile      = fullfile(folder, filenames{i});
        img_original = imread(imgFile);

        %-- Convert to double / grayscale --%
        img_double = im2double(img_original);
        img_gray   = rgb2gray(img_double);

        %-- Step 1: Binarize (adaptive Otsu) --%
        img_bw = imbinarize(img_gray, 'adaptive', 'Sensitivity', 0.3);

        %-- Step 2: Clean up --%
        img_bw = bwareaopen(img_bw, 100);
        img_bw = imfill(img_bw, 'holes');

        %-- Step 3: Region properties (includes SubarrayIdx for colour crop) --%
        stats = regionprops(img_bw, 'Area', 'Circularity', 'Eccentricity', ...
                                    'BoundingBox', 'SubarrayIdx');

        if ~isempty(stats)
            [~, idx] = max([stats.Area]);

            %-- Shape features --%
            circularity = stats(idx).Circularity;
            eccentricity = stats(idx).Eccentricity;

            %-- Colour features from the bounding-box crop --%
            if ~isempty(stats(idx).SubarrayIdx)
                subIdx      = stats(idx).SubarrayIdx;
                img_sub     = img_original(subIdx{1}, subIdx{2}, :);
                img_sub_dbl = im2double(img_sub);

                R = img_sub_dbl(:,:,1);
                G = img_sub_dbl(:,:,2);
                B = img_sub_dbl(:,:,3);

                meanR = mean(R(:));
                meanG = mean(G(:));
                meanB = mean(B(:));

                total_color   = meanR + meanG + meanB + eps;
                normR         = meanR / total_color;
                normG         = meanG / total_color;
                normB         = meanB / total_color;
                red_dominance = (meanR - meanB) * 2;

                feature_vector = [circularity, normR, normG, normB, ...
                                  red_dominance, eccentricity];
            else
                % Fallback: neutral colour, zero shape features
                feature_vector = [0, 1/3, 1/3, 1/3, 0, 0];
            end
        else
            % No region detected — return neutral defaults
            feature_vector = [0, 1/3, 1/3, 1/3, 0, 0];
        end

        features = [features; feature_vector]; %#ok<AGROW>
    end
end
