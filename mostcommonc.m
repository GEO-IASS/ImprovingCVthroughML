function W = mostcommonc(a)

if nargin < 1 | isempty(a)
    W = mapping(mfilename);
    W = setname(W,'Mostcommon');
    return
end

m=size(a,2);
mostcommonclass=mode(getlabels(a));
W = mapping('mostcommon_map','trained',mostcommonclass,getlablist(a),m,size(getlablist(a)));