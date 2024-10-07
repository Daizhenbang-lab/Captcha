function I2 = data_preprocessing(I)
    I = rgb2gray(I);
        
    %Fourier transform and denoise on fourier 
    F = fft2(I);
    F1 = fftshift(F);
    F1(157,204) = 0;
    F1(160,211) = 0;
    F1(166,225) = 0;
    F1(169,232) = 0;
    
    %recover image
    F_unshifted = ifftshift(F1);
    reconstructed_img = ifft2(F_unshifted);
    I = real(reconstructed_img);
    I = uint8(I);

    % smoothen the image 
    I = imgaussfilt(I,2);

    
    % Morphology to get rid of lines
    I = ~imbinarize(I);
    
    I2 = imerode(I, strel('disk',4));
    I2 = bwareaopen(I2, 400); 
    I2 = imdilate(I2, strel('disk',3));
    
    whitePixelsPercent = (nnz(I2)/numel(I2))*100;

    % check if after morphology too much information was removed 
    if whitePixelsPercent < 15
        I2 = imerode(I, strel('disk',1));
        I2 = bwareaopen(I2, 400); 
        I2 = imdilate(I2, strel('disk',3));
    end

    I2 = imerode(I2, strel('disk',5));
    I2 = bwareaopen(I2, 400); 
    I2 = imdilate(I2, strel('disk',3));
end