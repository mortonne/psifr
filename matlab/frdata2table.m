function tab = frdata2table(frdata, extra, names)
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
%  extra - cell array of strings
%      (optional) Cell array of strings indicating additional fields to include.
%      These fields must be in substructs .pres and .rec, giving the value of
%      that field for presentation and recall events, respectively.
%
%  names - cell array of strings
%      (optional) New name for each column in extra. If empty, the old name will
%      be used.
%
%  OUTPUTS
%  tab - table
%      Table of data in long format.

if nargin < 3
    names = {};
    if nargin < 2
        extra = {};
    end
end

unique_subject = unique(frdata.subject);
tab = [];
for i = 1:length(unique_subject)
    subj_data = trial_subset(frdata.subject == unique_subject(i), frdata);
    subj_tab = subj_frdata2table(subj_data, extra, names);
    tab = [tab; subj_tab];
end
