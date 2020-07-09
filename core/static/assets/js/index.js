$(function(){
    $('.nav-item').removeClass("active")
    $('.nav-item a[href="' + location.pathname + '"').parent().addClass('active')

    $('.data-remove-btn').on('click', function() {
        const datasetId = $(this).data('dataset-id')

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
            callback: function (result) {
                if (result)
                    $.get('/data/delete/' + datasetId).then(function() {
                        location.reload()
                    })
            }
        });
    })
});