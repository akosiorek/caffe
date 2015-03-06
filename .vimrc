set tabstop=4
set softtabstop=4
set shiftwidth=4
set noexpandtab
set colorcolumn=110
highlight ColorColumn ctermbg=darkgray

set makeprg=make\ -C\ build\ -j8
nnoremap <F4> :make!<cr>
