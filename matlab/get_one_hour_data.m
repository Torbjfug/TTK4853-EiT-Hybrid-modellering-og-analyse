function data_vector = get_one_hour_data(year, month, day, hour)
    air_potential_temperature_ml = get_data(year, month, day, 'air_potential_temperature_ml', 40, 50, 40, 50, 30, 40, hour, hour);
    x_wind_ml =  get_data(year, month, day, 'x_wind_ml', 40, 50, 40, 50, 30, 40, hour, hour);
    y_wind_ml = get_data(year, month, day, 'y_wind_ml', 40, 50, 40, 50, 30, 40, hour, hour);
    upward_air_velocity_ml =  get_data(year, month, day, 'upward_air_velocity_ml', 40, 50, 40, 50, 30, 40, hour, hour);
    air_pressure_ml =  get_data(year, month, day, 'air_pressure_ml', 40, 50, 40, 50, 30, 40, hour, hour);
    turbulence_index_ml =  get_data(year, month, day, 'turbulence_index_ml', 40, 50, 40, 50, 30, 40, hour, hour);
    turbulence_dissipation_ml =  get_data(year, month, day, 'turbulence_dissipation_ml', 40, 50, 40, 50, 30, 40, hour, hour);

    data_vector = [air_potential_temperature_ml(:); x_wind_ml(:); y_wind_ml(:); upward_air_velocity_ml(:); air_pressure_ml(:); turbulence_index_ml(:); turbulence_dissipation_ml(:)];    
    
end
