function K = feature_extraction(I)
    K = [];
    
    % check if we can rotate image to get better digits
    s = regionprops(I,'BoundingBox', 'Orientation');
    angle = 0;

    minBoundingBox = [];
    minArea = inf;
    for i = 1:length(s)
        bbox = s(i).BoundingBox;
        orientation = s(i).Orientation;

        allBoundingBoxes = cat(1, s.BoundingBox);

        % Find the minimum and maximum coordinates of bounding boxes
        minX = min(allBoundingBoxes(:, 1));
        minY = min(allBoundingBoxes(:, 2));
        maxX = max(allBoundingBoxes(:, 1) + allBoundingBoxes(:, 3));
        maxY = max(allBoundingBoxes(:, 2) + allBoundingBoxes(:, 4));

        if minArea > (maxX-minX)*(maxY-minY)
            minArea = (maxX-minX)*(maxY-minY);
            minBoundingBox = [minX, minY, maxX - minX, maxY - minY];
        end


        % Calculate the rotated bounding box
        rotatedBox = regionprops(imrotate(I, -orientation), 'BoundingBox');
        rotatedBoundingBox = cat(1, rotatedBox.BoundingBox);

        minX2 = min(rotatedBoundingBox(:, 1));
        minY2 = min(rotatedBoundingBox(:, 2));
        maxX2 = max(rotatedBoundingBox(:, 1) + rotatedBoundingBox(:, 3));
        maxY2 = max(rotatedBoundingBox(:, 2) + rotatedBoundingBox(:, 4));

        minRotatedBoundingBox = [minX2, minY2, abs(maxX2 - minX2), abs(maxY2 - minY2)];

        rotatedArea = (maxX2-minX2)*(maxY2-minY2);


        % Check if the rotated bounding box has a smaller area


        if rotatedArea < minArea
            minBoundingBox = minRotatedBoundingBox;
            minArea = rotatedArea;
            angle = orientation;
        end
    end

    % if the rotated image had a smaller bounding box then rotate the image
    % and crop it

    J = I;
    if angle ~= 0
        if minBoundingBox(3)>minBoundingBox(4)
            J = imcrop(imrotate(I,-angle), minBoundingBox);
        end

    end
    

    % get the three digits from image
    cc = bwconncomp(J,4);
    region = regionprops(cc, 'Image', 'Area');

    if cc.NumObjects == 1
        [h,w] = size(region(1).Image);
        split = round(w/3);
        d1 = imcrop(region(1).Image,[0 0 split h]);
        d2 = imcrop(region(1).Image,[split 0 split h]);
        d3 = imcrop(region(1).Image,[split*2 0 split h]);
        f1 = Features(d1);
        f2 = Features(d2);
        f3 = Features(d3);
        K(1,:,:) = f1;
        K(2,:,:) = f2;
        K(3,:,:) = f3;
        
    elseif cc.NumObjects == 2
        [h1,w1] = size(region(1).Image);
        [h2,w2] = size(region(2).Image);
        if w1 > w2 
            d1 = imcrop(region(1).Image,[0 0 round(w1/2) h1]);
            d2 = imcrop(region(1).Image,[round(w1/2) 0 round(w1/2) h1]);
            d3 = region(2).Image;
        else 
            d1 = region(1).Image;
            d2 = imcrop(region(2).Image,[0 0 round(w2/2) h2]);
            d3 = imcrop(region(2).Image,[round(w2/2) 0 round(w2/2) h2]);
        end
        f1 = Features(d1);
        f2 = Features(d2);
        f3 = Features(d3);
        K(1,:,:) = f1;
        K(2,:,:) = f2;
        K(3,:,:) = f3;
        
    elseif cc.NumObjects >= 3
        % sort Areas by size decending and pick largest three areas
        T = struct2table(region); 
        sortedT = sortrows(T, 'Area','descend'); % sort the table by value
        region = table2struct(sortedT);

        d1 = region(1).Image;
        d2 = region(2).Image;
        d3 = region(3).Image;
        K(1,:,:) = Features(d1);
        K(2,:,:) = Features(d2);
        K(3,:,:) = Features(d3);
    end
end

function F=Features(I)
    F1 = hu_moments(I);

    fts={'Circularity','Area','Centroid','Orientation','Solidity','Extent', 'EulerNumber','Eccentricity','Perimeter'}; 
	Ft=regionprops('Table',I,fts{:});
	[~,idx]=max(Ft.Area);
	F=[Ft(idx,:).Variables];
    
    F = cat(2,F,F1);

end

function H=hu_moments(I)
    [i,j,v]=find(I);
    if isrow(i)
	    i=i(:);j=j(:);v=v(:); 
    end
    
    %Zero moment
    mu00=sum(v); %M_00=mu_00
    ctr=sum([i,j].*v,1)./mu00;
    
    %Centralize
    i=i-ctr(1);
    j=j-ctr(2);
    
    %Scale invariant central moments of order {2,3}
    eta=nan(4,4);
    for p=0:3
	    for q=max(0,2-p):3-p %Only compute when needed
		    eta(p+1,q+1)=sum(i.^p .* j.^q .* v) / mu00.^(1+(p+q)/2);
	    end
    end
    
    %Moment invariants
    H(1)=eta(3,1)+eta(1,3);
    H(2)=(eta(3,1)-eta(1,3))^2 + 4*eta(2,2)^2;
    H(3)=(eta(4,1)-3*eta(2,3))^2 + (3*eta(3,2)-eta(1,4))^2;
    H(4)=(eta(4,1)+eta(2,3))^2 + (eta(3,2)+eta(1,4))^2;
    H(5)=(eta(4,1)-3*eta(2,3))* (eta(4,1)+eta(2,3))*( (eta(4,1)+eta(2,3))^2 - 3*(eta(3,2)+eta(1,4))^2 ) + ...
        (3*eta(3,2)-eta(1,4)) * (eta(3,2)+eta(1,4))*( 3*(eta(4,1)+eta(2,3))^2 - (eta(3,2)+eta(1,4))^2 );
    H(6)=(eta(3,1)-eta(1,3))*( (eta(4,1)+eta(2,3))^2 - (eta(3,2)+eta(1,4))^2 ) + 4*eta(2,2)*(eta(4,1)+eta(2,3))*(eta(3,2)+eta(1,4));
    H(7)=(3*eta(3,2)-eta(1,4))* (eta(4,1)+eta(2,3))*( (eta(4,1)+eta(2,3))^2 - 3*(eta(3,2)+eta(1,4))^2 ) - ...
         (eta(4,1)-3*eta(2,3))* (eta(3,2)+eta(1,4))*( 3*(eta(4,1)+eta(2,3))^2 - (eta(3,2)+eta(1,4))^2 );
    H(8)=eta(2,2)*( (eta(4,1)+eta(2,3))^2 - (eta(3,2)+eta(1,4))^2 ) - (eta(3,1)-eta(1,3))*(eta(4,1)+eta(2,3))*(eta(3,2)+eta(1,4));
end
