folder = 'data/train/'

for year = 2018
%     year
    for month = 1:10
        for day = 1:31
            disp([month,day])
            try
                filename = sprintf(folder+"%d_%02d_%02d.mat",year,month,day);
                load(filename)
                save(filename,'x_wind_ml','y_wind_ml','upward_air_velocity_ml','air_pressure_ml','-v7.3') 
                clear x_wind_ml y_wind_ml upward_air_velocity_ml air_pressure_ml
            catch error
                disp(error)
            end
            
            
        end
    end
end
