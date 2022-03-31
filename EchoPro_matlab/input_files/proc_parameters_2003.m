function  proc_parameters_2003
%% this file is to provide all required input filenames & some process parameter settings
%%
%% Written by Dezhang Chu    Written by Dezhang Chu, NOAA Fisheries, NWFSC
%% Last Modification:       10/02/2015

global para data 

% para.proc.source=2;    % CAN data only

%% Biological trawl data
switch para.bio_data_type
    case 1   % Acoustic and Trawl survey data
        para.bio.filename.trawl=['' para.data_root_dir 'Biological\2003\CAN\biodata_haul.xls'];
        para.bio.filename.gear=['' para.data_root_dir 'Biological\2003\CAN\biodata_gear.xls'];
        para.bio.filename.catch=['' para.data_root_dir 'Biological\2003\CAN\biodata_catch.xls'];
        para.bio.filename.length=['' para.data_root_dir 'Biological\2003\CAN\biodata_length.xls'];
        para.bio.filename.specimen=['' para.data_root_dir 'Biological\2003\CAN\biodata_specimen.xls'];
        para.bio.filename.trawl_US=[];
        para.bio.filename.gear_US=[];
        para.bio.filename.catch_US=[];
        para.bio.filename.length_US=[];
        para.bio.filename.specimen_US=[];
        para.bio.filename.trawl_CAN=[];
        para.bio.filename.gear_CAN=[];
        para.bio.filename.catch_CAN=[];
        para.bio.filename.length_CAN=[];
        para.bio.filename.specimen_CAN=[];
    case 2   % US Bottom Trawl data
        fprintf('******* Bottom Trawl Data are not available *******\n\n')
    case 3 % US observer trawl data
        para.bio.filename.trawl=['' para.data_root_dir 'Observer Data\Hake_Trawl_Chu_2003.xlsx'];
        para.bio.filename.gear=[''];
        para.bio.filename.catch=['' para.data_root_dir 'Observer Data\Hake_Catch_Chu_2003.xlsx'];
        para.bio.filename.length=['' para.data_root_dir 'Observer Data\Hake_Length_Chu_2003.xlsx'];
        para.bio.filename.specimen=['' para.data_root_dir 'Observer Data\Hake_Age_Chu_2003.xlsx'];
end
%% Stratification files
para.bio_acoust.filename.Transect_region_haul=['' para.data_root_dir 'Stratification\2003\US&CAN_T_reg_haul_final.xlsx'];
para.acoust.filename.strata=['' para.data_root_dir 'Stratification\2003\US&CAN strata 2003.xlsx'];
para.proc.stratification_filename=['' para.data_root_dir 'Stratification\2003\Stratification_geographic_Lat_2003.xlsx'];

%% NASC data
para.proc.transect_info_filename=['' para.data_root_dir 'Kriging files & parameters\Kriging grid files\Transect Bounds to 2011.xlsx'];                           % ST, BT, RT,and ET of transect for removing extra zeros 
para.acoust.filename.processed_data=['' para.data_root_dir 'Exports\US&CAN_detailsa_2003_table2y+_ALL_final.xlsx'];

%% kriging related files
data.in.filename.smoothed_contour=['' para.data_root_dir 'Kriging files & parameters\Kriging grid files\Smoothing_EasyKrig.xlsx'];
data.in.filename.grid_cell=['' para.data_root_dir 'Kriging files & parameters\Kriging grid files\krig_grid2_5nm_cut_centroids_2013.xlsx'];                         % 2013 cell res = 2.50 nmi with extended area coverage
para.krig.vario_krig_para_filename=['' para.data_root_dir 'Kriging files & parameters\2003\default_vario_krig_settings_final.xlsx'];
% para.krig.vario_krig_para_filename=['' para.data_root_dir 'Kriging files & parameters\2003\default_vario_krig_settings_orig.xlsx'];

% old parameter files
% para.bio_acoust.filename.Transect_region_haul=['' para.data_root_dir 'Stratification\2003\US&CAN_T_reg_haul_final_orig.xlsx'];
% para.proc.stratification_filename=['' para.data_root_dir 'Stratification\2003\Stratification_geographic_Lat_2003_orig.xlsx'];
% para.krig.vario_krig_para_filename=['' para.data_root_dir 'Kriging files & parameters\2003\default_vario_krig_settings_orig.xlsx'];

para.proc.ST_BT_RT_ET_zero_removal_flag=1;      % 0 = not remove zeros before ST and after ET; 1 = remove zeros before ST and after ET
para.proc.stratification_index=1;               % index for the chosen stratification
                                                % 1 = KS (trawl)-based, 2-8 = geographically based but close to trawl-based stratification
                                                % 0 = INPFC strata
para.proc.start_transect=1;                     % start transect number
para.proc.end_transect=144;                     % end transect number
para.proc.transect_offset=0;                    % transect offset added to the CAN transect when merge the uS and CAN data
para.proc.age1_haul=[ ];                        % trawls to be excluded if age-1 is excluded
para.proc.KS_stratification=1;                  % 1 - stratification based on KS (or trawl) - based analysis
                                                % 0 - geographically defined strata
para.bio.haul_no_offset=0;                      % Canadian's trawl number offset
para.bio.CAN_strata_num0=[];                    % for combined satrta definiation file
para.bio.database_type='Oracle';                % biodata format: 'Oracle' or 'FSCS'
para.acoust.TS_station_num=1;                   % number of trawl sampling stations, whose data are used to compute the TS
