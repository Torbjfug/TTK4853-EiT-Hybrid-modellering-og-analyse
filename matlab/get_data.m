%% Data types:
% 'time': Array of 64 bit Reals [time = 0..12]
% 'forecast_reference_time': 64 bit Real
% 'l': Array of 32 bit Reals [l = 0..40]
% 'rotated_latitude_longitude': 32 bit Integer
% 'x': Array of 32 bit Reals [x = 0..134]
% 'y': Array of 32 bit Reals [y = 0..135]
% 'longitude': Grid
% 'latitude': Grid
% 'geopotential_height_ml': Grid
% 'surface_altitude': Grid
% 'air_potential_temperature_ml': Grid
% 'x_wind_ml': Grid
% 'y_wind_ml': Grid
% 'upward_air_velocity_ml': Grid
% 'x_wind_10m': Grid
% 'y_wind_10m': Grid
% 'air_pressure_ml': Grid
% 'surface_air_pressure': Grid
% 'turbulence_index_ml': Grid
% 'turbulence_dissipation_ml': Grid

function data = get_data(year, month, day,  data_type, x_begin, x_end, y_begin, y_end, height_begin, height_end, time_begin, time_end)
    url = sprintf('https://thredds.met.no/thredds/dodsC/opwind/%d/%02d/%02d/simra_BESSAKER_%d%02d%02dT00Z.nc?time[%d:1:%d],forecast_reference_time,l[%d:1:%d],rotated_latitude_longitude,x[%d:1:%d],y[%d:1:%d],longitude[%d:1:%d][%d:1:%d],latitude[%d:1:%d][%d:1:%d],geopotential_height_ml[%d:1:%d][%d:1:%d][%d:1:%d][%d:1:%d],surface_altitude[%d:1:%d][%d:1:%d],air_potential_temperature_ml[%d:1:%d][%d:1:%d][%d:1:%d][%d:1:%d],x_wind_ml[%d:1:%d][%d:1:%d][%d:1:%d][%d:1:%d],y_wind_ml[%d:1:%d][%d:1:%d][%d:1:%d][%d:1:%d],upward_air_velocity_ml[%d:1:%d][%d:1:%d][%d:1:%d][%d:1:%d],x_wind_10m[%d:1:%d][%d:1:%d][%d:1:%d],y_wind_10m[%d:1:%d][%d:1:%d][%d:1:%d],air_pressure_ml[%d:1:%d][%d:1:%d][%d:1:%d][%d:1:%d],surface_air_pressure[%d:1:%d][%d:1:%d][%d:1:%d],turbulence_index_ml[%d:1:%d][%d:1:%d][%d:1:%d][%d:1:%d],turbulence_dissipation_ml[%d:1:%d][%d:1:%d][%d:1:%d][%d:1:%d]' ...
         , year, month, day, year, month, day, time_begin, time_end, height_begin, height_end,...
         x_begin, x_end, y_begin, y_end, y_begin, y_end, x_begin, x_end, y_begin, y_end, x_begin, x_end,...
         time_begin, time_end, height_begin, height_end, y_begin, y_end, x_begin, x_end,...
         y_begin, y_end, x_begin, x_end,...
         time_begin, time_end, height_begin, height_end, y_begin, y_end, x_begin, x_end,...
         time_begin, time_end, height_begin, height_end, y_begin, y_end, x_begin, x_end,...
         time_begin, time_end, height_begin, height_end, y_begin, y_end, x_begin, x_end,...
         time_begin, time_end, height_begin, height_end, y_begin, y_end, x_begin, x_end,...
         time_begin, time_end, y_begin, y_end, x_begin, x_end,...
         time_begin, time_end, y_begin, y_end, x_begin, x_end,...
         time_begin, time_end, height_begin, height_end, y_begin, y_end, x_begin, x_end,...
         time_begin, time_end, y_begin, y_end, x_begin, x_end,...
         time_begin, time_end, height_begin, height_end, y_begin, y_end, x_begin, x_end,...
         time_begin, time_end, height_begin, height_end, y_begin, y_end, x_begin, x_end);
    disp(data_type)
    data = ncread(url, data_type);
    if strcmp('air_pressure_ml',data_type)
        data(abs(data) > 2e9) = NaN;
    else
        data(abs(data) > 1e3) = NaN;
    end
end

