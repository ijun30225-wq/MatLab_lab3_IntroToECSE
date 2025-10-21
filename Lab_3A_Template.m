%% Lab 03 Part A - Image Processing Lab Matlab Script Template
%% Processes images to extract red features only. We are looking for stop signs.

close all;
clear all;

%% Step 1: Data Import
ds = imageDatastore("3A_traffic_sign_images","FileExtensions",".png");

% Read image #n - start with image #25, a clear stop sign
% Also try image #3 for a speed limit sign
% Try image #34 for a pedestrian crossing sign (not red)
n = 25;
for i = 1:n
    imRGB = read(ds);
end

figure
imshow(imRGB)
[~,filename,ext] = fileparts(ds.Files{n});
title([filename,ext])

% Flag for 8-bit images
if strcmpi(class(imRGB), 'uint8')
    eightBit = true;
else
    eightBit = false;
    imRGB = uint8(imRGB / 256); % convert to 8 bit
end

%% Step 2: Image Preprocessing
imLAB = rgb2lab(imRGB);
L = imLAB(:,:,1)/100;
L = adapthisteq(L,'NumTiles',[4 4],'ClipLimit',0.005);
imLAB(:,:,1) = L*100;
imRGB = lab2rgb(imLAB, 'OutputType','uint8');

%% Step 3: Separate into Color Bands and Plot Histograms
imgreen = imRGB(:,:,2);
imblue = imRGB(:,:,3);
imred = imRGB(:,:,1);
tiledlayout(1,4)
nexttile;
imshow(imRGB);
title("Original")
nexttile
imshow(imred)
title("Red Board")
nexttile
imshow(imgreen)
title("Green Board")
nexttile
imshow(imblue)
title("Blue Board")

figure;

% Create a tiled layout with 1 row and 4 columns
tiledlayout(1, 4);

% Plot the combined histogram with color coding
nexttile;
hold on; % Hold on to overlay histograms

% Plot histograms for each channel
histogram(imred, 'FaceColor', 'r', 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'Normalization', 'probability');
histogram(imgreen, 'FaceColor', 'g', 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'Normalization', 'probability');
histogram(imblue, 'FaceColor', 'b', 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'Normalization', 'probability');

% Add labels and title
xlabel('Intensity Value');
ylabel('Probability');
title('Combined Color Intensity Histogram');
hold off;

% Create individual histograms for each color band
nexttile;
imhist(imred);
title('Red Channel Histogram');

nexttile;
imhist(imgreen);
title('Green Channel Histogram');

nexttile;
imhist(imblue);
title('Blue Channel Histogram');



%% Step 4: Application of Masks to Identify Red Objects in Image


%% Step 5: Refinement of Red Objects in Image


%% Step 6: Feature Extraction for Classifying Stop Signs and Speed Limit Signs


%% Step 7: Edge Detection of Red Objects in Image


%% Step 8: Algorithm for determining if a sign in the data set is a stop sign