years = [2017,2018,2019]; 
days = [5,15];


for year = 2019
    year
    parfor month = 1:12
        month
        for day = 5:15:20
            data = get_data_day(year,month,day);
        end
    end
end
% 
% parfor month = 8:12
%     for day_index = 1:2
%         days = [5,15]
%         day = days(day_index);
%         month
%         get_data_day(2017,month,day);
%     end
% end
