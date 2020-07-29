$(function(){
    $('.nav-item').removeClass("active")
    $('.nav-item a[href="' + location.pathname + '"').parent().addClass('active')

    $('.data-remove-btn').on('click', function(e) {
        e.preventDefault()

        bootbox.confirm({
            message: "Do you really want to delete this dataset ?",
            buttons: {
                confirm: {
                    label: 'Yes',
                    className: 'btn-success'
                },
                cancel: {
                    label: 'No',
                    className: 'btn-danger'
                }
            },
            callback: (result) => {
                if (result) {
                    $.get($(this).attr('href')).then(function() {
                        location.reload()
                    })
                }                    
            }
        });
    })
});