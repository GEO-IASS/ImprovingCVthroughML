function W = minindexc(a)

if nargin < 1 | isempty(a)
    W = mapping(mfilename);
    W = setname(W,'Minindex');
    return
end

m=size(a,2);
W = mapping('minindex_map','trained',a,getlablist(a),m,m);

%m=size(a,2);
%for i=1:m
%    vec=zeros(1,m);
%    vec(i)=1;
%    points(i,:)=vec;
%end
%ds=dataset(points, (1:m)');
%W = knnc(ds,1);
%W = setname(W,'Minindex');
return