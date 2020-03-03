function data_vector = get_data_day(year, month, day)
    %x_begin = 1; x_end = 133; y_begin = 1; y_end = 133; l_begin = 1;l_end = 39;
    x_begin = 5; x_end = 130; y_begin = 5; y_end = 130; l_begin = 5;l_end = 35;

    size_vector = (x_end+1-x_begin)*(y_end+1-y_begin)*(l_end+1-l_begin);
    %air_potential_temperature_ml = get_data(year, month, day, 'air_potential_temperature_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    x_wind_ml =  get_data(year, month, day, 'x_wind_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    y_wind_ml = get_data(year, month, day, 'y_wind_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    upward_air_velocity_ml =  get_data(year, month, day, 'upward_air_velocity_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    %air_pressure_ml =  get_data(year, month, day, 'air_pressure_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    %turbulence_index_ml =  get_data(year, month, day, 'turbulence_index_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    %turbulence_dissipation_ml =  get_data(year, month, day, 'turbulence_dissipation_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);

    if any(isnan(x_wind_ml(:))) || any(isnan(y_wind_ml(:))) || any(isnan(upward_air_velocity_ml(:)))
        disp(month)
        disp(day)
    else
        filename = sprintf("data/validation/%d_%02d_%02d",year,month,day);
        save(filename,'x_wind_ml','y_wind_ml','upward_air_velocity_ml','-v7.3')
    end
    
    
    %save(filename,'air_potential_temperature_ml','x_wind_ml','y_wind_ml','y_wind_ml','upward_air_velocity_ml','air_pressure_ml','turbulence_index_ml','turbulence_dissipation_ml','-v7.3')
    
    data_vector = [%reshape(air_potential_temperature_ml,size_vector,13); 
                   reshape(x_wind_ml,size_vector,13);
                   reshape(y_wind_ml,size_vector,13);
                   reshape(upward_air_velocity_ml,size_vector,13);];
                   %reshape(air_pressure_ml, size_vector,13);
                   %reshape(turbulence_index_ml,size_vector,13);
                   %reshape(turbulence_dissipation_ml,size_vector,13)];    
end

