function structout = trial_subset(trials, structin, dim)
%TRIAL_SUBSET  Get a subset of each matrix in a structure.
%
%  For each vector or matrix in the input structure, this function
%  returns just the rows specified in the logical trials vector.
%
%  structout = trial_subset(trials, structin, dim);
%
%  INPUTS:
%    trials:  logical vector indicating rows of each subfield of
%             structin to keep.
%
%  structin:  structure of vectors and matrices. The number of rows in
%             each subfield must match the length of the trials vector,
%             or a subfield may contain a scalar value.
%
%       dim:  (optional) specifies the dimension of each subfield
%             to index. Default: 1.
%
%  OUTPUT:
%  structout:  structure of vectors and matrices, in the same
%              order as structin, but only containing the rows
%              specified in trials.

if nargin < 3
  dim = 1;
end

% sanity checks
if ~isstruct(structin)
  error('structin must be a structure.');
elseif ~islogical(trials) || ~isvector(trials)
  error('trials must be a logical array.');
end

structout = struct;
names = fieldnames(structin);

n_dims = max(structfun(@ndims, structin));
ind = repmat({':'}, 1, n_dims);
ind{dim} = trials;
for i = 1:length(names)
  this_field = structin.(names{i});
  if isstruct(this_field)
    % recurse for data.pres and data.rec and similar
    out_field = trial_subset(trials, this_field, dim);
  elseif isscalar(this_field) || ischar(this_field)
    % copy scalar values
    out_field = this_field;
  else
    message = sprintf('Field "%s" does not match index vector on dimension %d.', ...
                      names{i}, dim);
    assert(size(this_field, dim) == length(trials), message);
    out_field = this_field(ind{:});
  end

  structout.(names{i}) = out_field;
end
 

