years = [2017,2018,2019]; 
days = [5,15];

% 
% for year = 2018:2019
%     year
%     parfor month = 1:12
%         for day_index = 1:2
%             day = days(day_index);
%             month
%             get_data_day(year,month,day);
%         end
%     end
% end

% parfor month = 5:12
%     for day_index = 1:2
%         day = days(day_index);
%         month
%         get_data_day(2019,month,day);
%     end
% end
data = get_data_day(2019,1,2);