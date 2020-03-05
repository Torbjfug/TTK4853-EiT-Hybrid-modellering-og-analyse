function get_data_day(year, month, day,folder)
    %x_begin = 1; x_end = 133; y_begin = 1; y_end = 133; l_begin = 1;l_end = 39;
    x_begin = 2; x_end = 133; y_begin = 2; y_end = 133; l_begin = 7;l_end =38;

    %air_potential_temperature_ml = get_data(year, month, day, 'air_potential_temperature_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    x_wind_ml =  get_data(year, month, day, 'x_wind_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    y_wind_ml = get_data(year, month, day, 'y_wind_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    upward_air_velocity_ml =  get_data(year, month, day, 'upward_air_velocity_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    %air_pressure_ml =  get_data(year, month, day, 'air_pressure_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    %turbulence_index_ml =  get_data(year, month, day, 'turbulence_index_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    %turbulence_dissipation_ml =  get_data(year, month, day, 'turbulence_dissipation_ml', x_begin, x_end, y_begin, y_end, l_begin, l_end, 0, 12);
    if any(isnan(x_wind_ml(:))) || any(isnan(y_wind_ml(:))) || any(isnan(upward_air_velocity_ml(:)))
        error('Invalid data') 
    else
        filename = sprintf(folder+"%d_%02d_%02d",year,month,day);
        save(filename,'x_wind_ml','y_wind_ml','y_wind_ml','upward_air_velocity_ml','-v7.3')
    end
end

