// load file 
$('#choose-file-icon').on('click', function () {
    $('input[name=dataset]').click()
})

$('input[name=dataset]').on('change', function () {
    if ($(this).val() != '') $('#load-dataset-btn').removeAttr('disabled')
    else $('#load-dataset-btn').attr('disabled', 'disabled')
    // set filename
    $('#file-name').html($(this).val())
})

$('#load-dataset-btn').click(function () {
    if ($('input[name=dataset]')[0].files[0] != undefined) {
        $('#file-upload-spinner').show()
        $('#load-dataset-btn').attr('disabled', 'disabled')
        $('input[name=dataset]').attr('disabled', 'disabled')

        var formData = new FormData();
        formData.append('file', $('input[name=dataset]')[0].files[0]);
        
        $.ajax({
            url: 'upload.php',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (data) {
                $('#file-upload-spinner').hide()
                $('#load-dataset-btn').removeAttr('disabled')
                $('input[name=dataset]').removeAttr('disabled')
                $('#file-name').html('')

                
                $('input[name=dataset]').val(null)
                console.log(data);
                alert(data);
            },
            error: function(err) {
                $('#file-upload-spinner').hide()
                $('#load-dataset-btn').removeAttr('disabled')
                $('input[name=dataset]').removeAttr('disabled')
                $('#file-name').html('')

                
                $('input[name=dataset]').val(null)
                console.log(err);
                alert(err);
            }
        });
    }
})