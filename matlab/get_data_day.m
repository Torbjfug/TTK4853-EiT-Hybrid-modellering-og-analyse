function data_vector = get_data_day(year, month, day)
    x_begin = 0; x_end = 134; y_begin = 0; y_end = 135; l_begin = 0;l_end = 40;
    size_vector = (x_end+1-x_begin)*(y_end+1-y_begin)*(l_end+1-l_begin);
    air_potential_temperature_ml = get_data(year, month, day, 'air_potential_temperature_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    x_wind_ml =  get_data(year, month, day, 'x_wind_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    y_wind_ml = get_data(year, month, day, 'y_wind_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    upward_air_velocity_ml =  get_data(year, month, day, 'upward_air_velocity_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    air_pressure_ml =  get_data(year, month, day, 'air_pressure_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    turbulence_index_ml =  get_data(year, month, day, 'turbulence_index_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    turbulence_dissipation_ml =  get_data(year, month, day, 'turbulence_dissipation_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    data_vector = [reshape(air_potential_temperature_ml,size_vector,13); 
                   reshape(x_wind_ml,size_vector,13);
                   reshape(y_wind_ml,size_vector,13);
                   reshape(upward_air_velocity_ml,size_vector,13);
                   reshape(air_pressure_ml, size_vector,13);
                   reshape(turbulence_index_ml,size_vector,13);
                   reshape(turbulence_dissipation_ml,size_vector,13)];    
end

