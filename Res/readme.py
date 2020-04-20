# 已下载的第三方库# IPOPT的下载指导中，第三方库的说明有些问题，其中Blaz库下载后没有解压在对应文件夹中，make可能会失败
# # 另外要注意到如果先安装了clang(也就是先安装casadi，后安装IPOPT）,可能会Make失败，解决方法是在..configure 时添加
# #  coin_skip_warn_cxxflags=yes
# # ref: https://github.com/JuliaOpt/Ipopt.jl/issues/13
