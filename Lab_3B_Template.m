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

img_double = im2double(img_original);
R = img_double(:,:,1);
G = img_double(:,:,2);
B = img_double(:,:,3);
img_gray = rgb2gray(img_double);

mean_intensity = mean(img_gray(:));


combined_mask = zeros(size(img_gray));

if mean_intensity > 0.3
    if mean_intensity < 0.5
        img_darkened = img_gray .* (1 - 2*B);
    else
        img_darkened = img_gray .* (1 - 3*B);  
    end

    dark_mask = 1 - img_gray; 
    mask_darkened = img_darkened .* dark_mask;

    red_emphasis = R - (G + B)/2;
    red_emphasis(red_emphasis < 0) = 0;

    red_emphasis = 2 * red_emphasis;  

    combined = mask_darkened + red_emphasis;


    redMask = (R >= 60/255) & (R <= 1) & (B <= .5); 
    combined_mask = combined .* double(redMask);

    img_bw = imbinarize(combined_mask, 'adaptive', 'Sensitivity', 0.5);
    img_bw = bwareaopen(img_bw, 100);
    img_bw = imfill(img_bw, "holes");

    stats = regionprops(img_bw,'Area', "PixelIdxList",'Circularity','SubarrayIdx');
        [~, idx] = max([stats.Area]);
        circularity = stats.Circularity;
        subIdx = stats(idx).SubarrayIdx;
        img_sub = img_original(subIdx{1}, subIdx{2}, :);
        img_double = im2double(img_sub);
        R = img_double(:,:,1);
        G = img_double(:,:,2);
        B = img_double(:,:,3);
        totalMean = mean(R(:)) + mean(G(:)) + mean(B(:));
        meanR = mean(R(:)) / totalMean;
        meanG = mean(G(:)) / totalMean;
        meanB = mean(B(:)) / totalMean;
        
       features = [features;meanR,meanG,meanB ];
else 
    img_bw = imbinarize(img_gray);
    img_bw = bwareaopen(img_bw, 100); img_bw = imfill(img_bw, "holes");
    img_bw = imfill(img_bw, "holes");

 
    stats = regionprops(img_bw,'Area', "PixelIdxList",'Circularity','SubarrayIdx');
        [~, idx] = max([stats.Area]);
        circularity = stats.Circularity;
        subIdx = stats(idx).SubarrayIdx;
        img_sub = img_original(subIdx{1}, subIdx{2}, :);
        img_double = im2double(img_sub);
        R = img_double(:,:,1);
        G = img_double(:,:,2);
        B = img_double(:,:,3);
        totalMean = mean(R(:)) + mean(G(:)) + mean(B(:));
        meanR = mean(R(:)) / totalMean;
        meanG = mean(G(:)) / totalMean;
        meanB = mean(B(:)) / totalMean;
        
       features = [features;meanR,meanG,meanB];
end
    end
end


