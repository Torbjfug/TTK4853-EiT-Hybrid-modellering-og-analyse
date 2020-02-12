year = 2019; day = 5;
number_of_days = 28;
sprintf('https://thredds.met.no/thredds/dodsC/opwind/%d/0%d/0%d/simra_BESSAKER_%d0%d0%d0Z.nc', year, month,day, year, month, day);
data = zeros(752760*7, 13*number_of_days);
for i = 1:1
    for j = 1:number_of_days
        j
        data(:,(j)*13+1:(1+j)*13) = get_data_day(2019,i,j);
    end
end
a = get_one_hour_data(2019,4,7,1);