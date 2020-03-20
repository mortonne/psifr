function tab = subj_frdata2table(frdata, extra, names)
%SUBJ_FRDATA2TABLE   Export an FRdata struct for one subject to a table.
%
%  subj_frdata2table(frdata, extra)
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

if nargin < 3
    names = {};
    if nargin < 2
        extra = {};
    end
end

% compile basic information
[n_list, n_position] = size(frdata.pres_items);
max_recall = size(frdata.recalls, 2);
n_study = numel(frdata.pres_items);
n_recall = nnz(frdata.recalls ~= 0);
n_trial = n_study + n_recall;

% standard fields
trial_type = cell(n_trial, 1);
list = zeros(n_trial, 1);
position = zeros(n_trial, 1);
item = cell(n_trial, 1);

% extra custom fields
if ~isempty(extra)
    if isempty(names)
        names = repmat({''}, size(extra));
    end

    extra_vectors = struct();
    for i = 1:length(extra)
        if isempty(names{i})
            names{i} = extra{i};
        end
        f = names{i};
        extra_vectors.(f) = zeros(n_trial, 1);
    end
end

% prepare subject field
if isfield(frdata, 'subject')
    if iscell(frdata.subject)
        subject = cell(n_trial, 1);
    else
        subject = zeros(n_trial, 1);
    end
else
    subject = repmat({'n/a'}, [n_trial, 1]);
end
if length(unique(subject)) > 1
    error('Data struct has multiple subjects. Call frdata2table instead.')
end

% unpack trial information
ind = 1;
for i = 1:n_list
    % study trials
    for j = 1:n_position
        subject(ind) = frdata.subject(i);
        trial_type{ind} = 'study';
        list(ind) = i;
        position(ind) = j;
        item{ind} = frdata.pres_items{i, j};
        if ~isempty(extra)
            for k = 1:length(extra)
                f1 = extra{k};
                f2 = names{k};
                mat = frdata.pres.(f1);
                extra_vectors.(f2)(ind) = mat(i, j);
            end
        end

        ind = ind + 1;
    end

    % test trials
    for j = 1:max_recall
        recall = frdata.recalls(i, j);
        if recall == 0 || isnan(recall)
            % end of recall
            break
        end

        subject(ind) = frdata.subject(i);
        trial_type{ind} = 'recall';
        list(ind) = i;
        position(ind) = j;
        item{ind} = frdata.rec_items{i, j};
        if ~isempty(extra)
            for k = 1:length(extra)
                f1 = extra{k};
                f2 = names{k};
                mat = frdata.rec.(f1);
                extra_vectors.(f2)(ind) = mat(i, j);
            end
        end

        ind = ind + 1;
    end
end

tab = table(subject, list, position, trial_type, item);

if ~isempty(extra)
    for i = 1:length(names)
        f = names{i};
        tab.(f) = extra_vectors.(f);
    end
end
