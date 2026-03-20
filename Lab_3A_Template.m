%% Lab 03 Part A - Image Processing Lab Matlab Script
%% Processes images to extract red features only. We are looking for stop signs.
%
% Authors: Nicholas Gladu, Jun Iguchi
% Course:  ECSE 1010 - Introduction to ECSE, RPI
%
% Usage:
%   Set n to the 1-based index of the image you want to inspect, then run
%   the script. The 3A_traffic_sign_images/ folder must be on your MATLAB path.
%
% Suggested image indices:
%   Stop signs            : 7, 22, 23, 24, 25, 26, 27
%   Speed limit signs     : 3, 4, 5, 6, 8, 9, 10, 11, 21
%   Pedestrian/crosswalk  : 15, 16, 17, 18, 34

close all;
clear all;

%% -------------------------------------------------------------------------
%% Step 1: Data Import
%% -------------------------------------------------------------------------
ds = imageDatastore("3A_traffic_sign_images", "FileExtensions", ".png");

n = 27;  % <-- change this to select a different image

for i = 1:n
    imRGB = read(ds);
end

figure;
imshow(imRGB);
[~, filename, ext] = fileparts(ds.Files{n});
title([filename, ext]);

% Ensure image is 8-bit
if strcmpi(class(imRGB), 'uint8')
    eightBit = true;
else
    eightBit = false;
    imRGB = uint8(imRGB / 256);
end

%% -------------------------------------------------------------------------
%% Step 2: Image Preprocessing
%%   Convert to LAB, apply adaptive histogram equalisation, convert back.
%% -------------------------------------------------------------------------
imLAB = rgb2lab(imRGB);
L = imLAB(:,:,1) / 100;
L = adapthisteq(L, 'NumTiles', [4 4], 'ClipLimit', 0.005);
imLAB(:,:,1) = L * 100;
imRGB = lab2rgb(imLAB, 'OutputType', 'uint8');

%% -------------------------------------------------------------------------
%% Step 3: Separate into Color Bands and Plot Histograms
%% -------------------------------------------------------------------------
imRed   = imRGB(:,:,1);
imGreen = imRGB(:,:,2);
imBlue  = imRGB(:,:,3);

% --- Color band images ---
figure;
tiledlayout(1, 4);
nexttile; imshow(imRGB);   title("Original");
nexttile; imshow(imRed);   title("Red Band");
nexttile; imshow(imGreen); title("Green Band");
nexttile; imshow(imBlue);  title("Blue Band");

% --- Histograms ---
figure('Position', [100, 100, 1600, 400]);
tiledlayout(1, 4, 'TileSpacing', 'Compact', 'Padding', 'Compact');

nexttile;
hold on;
histogram(imRed,   'FaceColor', 'r', 'DisplayStyle', 'bar', 'EdgeColor', 'none');
histogram(imGreen, 'FaceColor', 'g', 'DisplayStyle', 'bar', 'EdgeColor', 'none');
histogram(imBlue,  'FaceColor', 'b', 'DisplayStyle', 'bar', 'EdgeColor', 'none');
hold off;
title('Combined RGB Histogram'); xlabel('Intensity'); ylabel('Pixel Count');
legend('Red', 'Green', 'Blue');

nexttile;
histogram(imRed,   'FaceColor', 'r');
title('Red Channel'); xlabel('Intensity'); ylabel('Pixel Count');

nexttile;
histogram(imGreen, 'FaceColor', 'g');
title('Green Channel'); xlabel('Intensity'); ylabel('Pixel Count');

nexttile;
histogram(imBlue,  'FaceColor', 'b');
title('Blue Channel'); xlabel('Intensity'); ylabel('Pixel Count');

%% -------------------------------------------------------------------------
%% Step 4: Build Color Masks
%%   Red mask  : high red intensity
%%   Blue/Green masks : used to subtract non-red areas (white text, sky, etc.)
%%   totalMask : red AND NOT blue AND NOT green
%% -------------------------------------------------------------------------
% --- Exploratory single-channel masks (shown for inspection) ---
figure;
redMask_init = (imRed >= 135) & (imRed <= 210);
imshow(redMask_init); title('Exploratory Red Mask (135–210)');

figure;
blueMask_init = (imBlue >= 120) & (imBlue <= 150);
imshow(blueMask_init); title('Exploratory Blue Mask (120–150)');

% --- Final threshold masks ---
redMask   = (imRed   >= 100) & (imRed   <= 255);
blueMask  = (imBlue  >= 100) & (imBlue  <= 255);
greenMask = (imGreen >= 120) & (imGreen <= 255);
totalMask = redMask & ~blueMask & ~greenMask;

figure;
tiledlayout(1, 4);
nexttile; imshow(redMask);   title('Red Mask');
nexttile; imshow(blueMask);  title('Blue Mask');
nexttile; imshow(greenMask); title('Green Mask');
nexttile; imshow(totalMask); title('Total Mask (Red only)');

%% -------------------------------------------------------------------------
%% Step 5: Morphological Refinement
%%   1. Remove small isolated bright regions (bwareaopen)
%%   2. Fill enclosed dark holes (imfill)
%%   3. Smooth edges with a disk-shaped closing element (imclose)
%% -------------------------------------------------------------------------
totalMask = bwareaopen(totalMask, 150);
totalMask = imfill(totalMask, 'holes');
se        = strel('disk', 50);
totalMask = imclose(totalMask, se);

figure;
imshow(totalMask);
title('Refined Total Mask');

%% -------------------------------------------------------------------------
%% Step 6: Apply Mask to Original Image
%% -------------------------------------------------------------------------
maskedRed   = uint8(imRed)   .* uint8(totalMask);
maskedGreen = uint8(imGreen) .* uint8(totalMask);
maskedBlue  = uint8(imBlue)  .* uint8(totalMask);
resultImage = cat(3, maskedRed, maskedGreen, maskedBlue);

figure;
imshow(resultImage);
title('Filtered RGB Image with Total Mask');

%% -------------------------------------------------------------------------
%% Step 7: Edge Detection (Sobel)
%% -------------------------------------------------------------------------
edgeImg = edge(totalMask, 'sobel');
figure;
imshow(edgeImg);
title('Edges of Red Object Mask');

%% -------------------------------------------------------------------------
%% Step 8: Rule-Based Classification
%%   Uses circularity (4πA/P²) and mean red intensity to distinguish:
%%     • Stop Sign      (high red, lower circularity — octagonal)
%%     • Speed Limit    (high red, higher circularity — circular)
%%     • Pedestrian     (high blue or no red region detected)
%% -------------------------------------------------------------------------
N_largest = 1;  % only consider the single largest region

Ltemp = bwlabel(totalMask);
props = regionprops(Ltemp, 'Area', 'Perimeter', 'Centroid', 'BoundingBox');
areas = [props.Area];

if isempty(areas)
    disp("Pedestrian Sign  (no red region detected)");
else
    [~, order]  = sort(areas, 'descend');
    keepIdx     = order(1:min(N_largest, length(order)));
    maskFiltered = ismember(Ltemp, keepIdx);

    L_filtered = bwlabel(maskFiltered);
    B          = bwboundaries(maskFiltered, 'noholes');
    stats      = regionprops(L_filtered, 'Area', 'Perimeter', 'Centroid', 'BoundingBox');

    imD = im2double(imRGB);
    Rch = imD(:,:,1);
    Gch = imD(:,:,2);
    Bch = imD(:,:,3);

    for k = 1:length(B)
        regionMask = (L_filtered == k);
        A = stats(k).Area;
        P = stats(k).Perimeter;

        if P == 0
            circ_value = 0;
        else
            circ_value = 4 * pi * A / (P^2);
        end

        meanR = mean(Rch(regionMask));
        meanG = mean(Gch(regionMask));
        meanB = mean(Bch(regionMask));

        isRed  = meanR > 0.5;
        isBlue = meanB > 0.5;

        if isRed
            if meanR > 0.7 || circ_value < 0.8
                label = "Stop Sign";
            else
                label = "Speed Limit Sign";
            end
        elseif isBlue
            label = "Pedestrian Sign";
        else
            label = "Other / unclassified";
        end

        fprintf('%s  |  circularity=%.3f  meanR=%.3f  meanB=%.3f\n', ...
                label, circ_value, meanR, meanB);
    end
end
