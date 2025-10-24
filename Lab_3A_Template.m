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
imGreen = imRGB(:,:,2);
imBlue = imRGB(:,:,3);
imRed = imRGB(:,:,1);
tiledlayout(1,4)
nexttile;
imshow(imRGB);
title("Original")
nexttile
imshow(imRed)
title("Red Board")
nexttile
imshow(imGreen)
title("Green Board")
nexttile
imshow(imBlue)
title("Blue Board")

figure;

figure('Position', [100, 100, 1600, 400]); 
tiledlayout(1, 4, 'TileSpacing', 'Compact', 'Padding', 'Compact');

nexttile;
hold on;
histogram(imRed, 'FaceColor', 'r', 'DisplayStyle', 'bar', 'EdgeColor', 'none');
histogram(imGreen, 'FaceColor', 'g', 'DisplayStyle', 'bar', 'EdgeColor', 'none');
histogram(imBlue, 'FaceColor', 'b', 'DisplayStyle', 'bar', 'EdgeColor', 'none');
hold off;
title('Combined Histogram of RGB Channels');
xlabel('Intensity');
ylabel('Pixel Count');
legend('Red Channel', 'Green Channel', 'Blue Channel');

% Red channel histogram
nexttile;
histogram(imRed, 'FaceColor', 'r');
title('Red Channel');
xlabel('Intensity');
ylabel('Pixel Count');

% Green channel histogram
nexttile;
histogram(imGreen, 'FaceColor', 'g');
title('Green Channel');
xlabel('Intensity');
ylabel('Pixel Count');

% Blue channel histogram
nexttile;
histogram(imBlue, 'FaceColor', 'b');
title('Blue Channel');
xlabel('Intensity');
ylabel('Pixel Count'); 
%% Step 4: Application of Masks to Identify Red Objects in Image
redMask = (imRed >= 145) & (imRed <= 210);
imshow(redMask);
blueMask = (imBlue >= 100) & (imBlue <= 150);
imshow(blueMask);
figure; 
tiledlayout(1, 4);
nexttile;
redMask = (imRed >= 145) & (imRed <= 255);
imshow(redMask);
title('Red Mask');
nexttile;
blueMask = (imBlue >= 100) & (imBlue <= 255);
imshow(blueMask);
title('Blue Mask');
nexttile;
GreenMask = (imGreen >= 100) & (imGreen <= 255);
imshow(GreenMask);
title('Green Mask');
nexttile;
totalMask = redMask-blueMask-GreenMask;
imshow(totalMask);

%% Step 5: Refinement of Red Objects in Image
TotalMask = bwareaopen(totalMask,150);
se = strel('disk',50);
TotalMask = imfill(TotalMask,"holes");
TotalMask = imclose(TotalMask,se);
imshow(TotalMask)

%% Step 6: Feature Extraction for Classifying Stop Signs and Speed Limit Signs
maskedRed = uint8(imRed) .* uint8(TotalMask);   % Apply total mask to red channel
maskedGreen = uint8(imGreen) .* uint8(TotalMask); % Set green channel to zero where red is present
maskedBlue = uint8(imBlue) .* uint8(TotalMask); % Set blue channel to zero where red is present

% Step 4: Concatenate the masked color channels into an RGB image
resultImage = cat(3, maskedRed, maskedGreen, maskedBlue);

% Step 5: Display the resulting image
imshow(resultImage);
title('Filtered RGB Image with Total Mask');


%% Step 7: Edge Detection of Red Objects in Image
edge = edge(TotalMask,"sobel");
imshow(edge);
title('Edge of Red Object Mask');

%% Step 8: Algorithm for determining if a sign in the data set is a stop sign
avg = mean2(resultImage(:,:,1));
if avg > 30 
    disp('This image is likely a stop sign.');
else
    disp('This image is not a stop sign.');
end
[B,L] = bwboundaries(TotalMask, 'noholes');



