years = [2017,2018,2019]; 
days = [5,15];

% 
% for year = 2018
%     year
%     for month = 11:12
%         parfor day = 1:31
%             disp([month,day])
%             try
%                 get_data_day(year,month,day,"data/train/");
%             catch
%                 disp("error")
%                 disp([year,month,day]) 
%             end
%         end
%     end
% end

for year = 2019
    year
    parfor month = 1:12
        for day = 5:15:20
            month
            try
                get_data_day(year,month,day,"data/validation/");
            catch
                disp([year,month,day])
                try
                    get_data_day(year,month,day+1,"data/validation/");
                catch
                   disp([year,day+1,month]) 
                end
            end
            
        end
    end
end
