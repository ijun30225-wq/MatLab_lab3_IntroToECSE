%% Lab 03 Part A - Image Processing Lab Matlab Script Template
%% Processes images to extract red features only. We are looking for stop signs.

close all;
clear all;

%% Step 1: Data Import
ds = imageDatastore("3A_traffic_sign_images","FileExtensions",".png");

% Read image #n - start with image #25, a clear stop sign7, 22, 23, 24, 26,
% Also try image #3 for a speed limit sign4, 5, 6, 8, 9, 10, 11, 21
% Try image #34 for a pedestrian crossing sign (not red) 15, 16, 17, 18
n = 27;
% 3~ 
% wrong

for i = 1:n
    imRGB = read(ds);
end

figure;
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

figure;
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
figure;
redMask = (imRed >= 135) & (imRed <= 210);
imshow(redMask);
title('Initial Red Mask')

figure;
blueMask = (imBlue >= 120) & (imBlue <= 150);
imshow(blueMask);
title('Initial Blue Mask')

figure; 
tiledlayout(1, 4);
nexttile;
redMask = (imRed >= 100) & (imRed <= 255);
imshow(redMask);
title('Red Mask');
nexttile;
blueMask = (imBlue >= 100) & (imBlue <= 255);
imshow(blueMask);
title('Blue Mask');
nexttile;
GreenMask = (imGreen >= 120) & (imGreen <= 255);
imshow(GreenMask);
title('Green Mask');
nexttile;
totalMask = redMask & ~ blueMask & ~ GreenMask;
imshow(totalMask);
title('Total Mask')

%% Step 5: Refinement of Red Objects in Image
totalMask = bwareaopen(totalMask,150);
se = strel('disk',50);
totalMask = imfill(totalMask,"holes");
totalMask = imclose(totalMask,se);

figure;
imshow(totalMask)
title('Refined Total Mask')

%% Step 6: Feature Extraction for Classifying Stop Signs and Speed Limit Signs
maskedRed = uint8(imRed) .* uint8(totalMask);
maskedGreen = uint8(imGreen) .* uint8(totalMask);
maskedBlue = uint8(imBlue) .* uint8(totalMask);

resultImage = cat(3, maskedRed, maskedGreen, maskedBlue);

figure;
imshow(resultImage);
title('Filtered RGB Image with Total Mask');

%% Step 7: Edge Detection of Red Objects in Image
edgeImg = edge(totalMask,"sobel");
figure;
imshow(edgeImg);
title('Edge of Red Object Mask');

%% Step 8: Algorithm for determining if a sign in the data set is a stop sign
N_largest = 1;

Ltemp = bwlabel(totalMask);
props = regionprops(Ltemp,'Area','Perimeter','Centroid','BoundingBox');
areas = [props.Area];

if isempty(areas)
    disp("Pedestrian Sign");
else
    [~, order] = sort(areas, 'descend');
    keepIdx = order(1:min(N_largest, length(order)));
    maskFiltered = ismember(Ltemp, keepIdx);
    
    L_filtered = bwlabel(maskFiltered);
    B = bwboundaries(maskFiltered, 'noholes');
    stats = regionprops(L_filtered, 'Area', 'Perimeter', 'Centroid', 'BoundingBox');
    
    imD = im2double(imRGB);
    Rch = imD(:,:,1);
    Gch = imD(:,:,2);
    Bch = imD(:,:,3);
    HSV = rgb2hsv(imD);
    Hch = HSV(:,:,1);
    Sch = HSV(:,:,2);
    Vch = HSV(:,:,3);
    
    colorFactor = 1.2;
    minSat = 0.25;
    threshold = 0.80;
    
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
        meanH = mean(Hch(regionMask));
        meanS = mean(Sch(regionMask));
        meanV = mean(Vch(regionMask));
        
        isRed = meanR > .5;
        isBlue = meanB > .5;
        
        if isRed
            if meanR > .7 || circ_value < .8
                label = "Likely STOP sign (red + polygonal)";
            else
                label = "Circular red sign (e.g. speed-limit border)";
            end
        elseif isBlue
            label = "Blue sign (e.g. pedestrian)";
        else
            label = "Other / not red/blue";
        end
        
        disp( label + "  C=" + num2str(circ_value) + meanR + meanB);
    end
end






