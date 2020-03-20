function tab = frdata2table(frdata, extra)
%FRDATA2TABLE   Export an FRdata struct with multiple subjects to a table.
%
%  tab = frdata2table(frdata, extra)
%
%  INPUTS
%  frdata - free recall data structure
%      Standard free recall data structure used by EMBAM. Must have the
%      following fields:
%          pres_items - cell array of strings with presented items
%          rec_items  - cell array of strings with recalled items
%          recalls    - numeric array where zero or nan indicates no recall
%
%  extra - struct
%      (optional) Cell array of strings indicating additional fields to include.
%      These fields must be in substructs .pres and .rec, giving the value of
%      that field for presentation and recall events, respectively.
%
%  OUTPUTS
%  tab - table
%      Table of data in long format.

unique_subject = unique(frdata.subject);
tab = [];
for i = 1:length(unique_subject)
    subj_data = trial_subset(frdata.subject == unique_subject(i), frdata);
    subj_tab = subj_frdata2table(subj_data);
    tab = [tab; subj_tab];
end
