load('data/calibration/2019_04_15.mat');

I = find(isnan(upward_air_velocity_ml));
sizes = size(upward_air_velocity_ml);
[x,y,z,t] = ind2sub(sizes,I);


files = dir('data/*.mat');
corrupt_wind = {};

for file = {files.name}
    filename = "data/" + cell2mat(file)
    load(filename)
    v = upward_air_velocity_ml;
    
    
    v(abs(v) > 1e10) = NaN;
    
    a = sum(isnan(v(:)));
    b = sum(~isnan(v(:)));
    if a>0
        corrupt_wind{end+1} = filename;
        2;
    end
    
    
end
% nanmin(v(:))
% nanmax(v(:))
% 
% 
% 
% nanmean(v(:))
% nanmin(v(:))
histogram(v(:))