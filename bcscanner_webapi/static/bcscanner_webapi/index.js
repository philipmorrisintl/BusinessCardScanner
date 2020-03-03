

Vue.component('ocr-chip', {
    template: "#ocr-chip-template",
    delimiters: ['[[', ']]'],

    props: {
        removable: {type: Boolean, default: false},
        undraggable: {type: Boolean, default: false},
        value: {type: String, default: ""},
    },

    data () {
        return {
            editedValue: "",
        }
    },

    created () {
        this.editedValue = this.value;
    },

    mounted () {
        Stretchy.resize(this.$refs.input);
    },

    methods: {
        handleInput () {
            this.$emit('input', this.editedValue);
        },

        handleInputKeyPress (event) {
            if(event.keyCode === 13) {
                event.preventDefault();
            }
        },
    }


});

Vue.component('ocr-select', {

    template: '#ocr-select-template',
    delimiters: ['[[', ']]'],

    props: {
        label: {type: String, default: ""},
        value: {type: Array, default: () => []},
        options: {type: Array, default: () => []},
        group: {type: String, default: "ocr"},
        recycleBin: {type: Object, default: null},
        clearable: {type: Boolean, default: false},
        undraggable: {type: Boolean, default: false},
        lineItems: {type: Boolean, default: false},
        rules: {type: Array, default: () => []},
    },

    data () {
        return {
            filteredOptions: [],
            editedValue: "",
            objectValues: [],
            unwatchObjects: false,
            unwatchValue: false,
            popupVisible: false,
        }
    },

    computed : {

        stackLabel () {
            return this.value && this.value.length > 0;
        }

    },

    watch: {
        objectValues () {
            if(this.unwatchObjects) {
                this.unwatchObjects = false;
                return;
            }
            this.value.splice(0, this.value.length);
            for(obj of this.objectValues) {
                this.value.push(obj.name);
            }
        },

        value () {
            if(this.unwatchValue) {
                this.unwatchValue = false;
                return;
            }
            this.unwatchObjects = true;
            this.objectValues.splice(0, this.objectValues.length);
            for(val of this.value) {
                this.objectValues.push({id: Quasar.utils.uid(), name: val});
            }
            this.$emit('input', this.value);
        },
    },

    created () {
        this.filteredOptions = this.options;

        for(val of this.value) {
            this.objectValues.push({id: Quasar.utils.uid(), name: val});
        }
    },

    methods: {

        handleInput (newValue) {
            this.$emit('input', newValue);
        },

        clearContent () {
            var cleared = this.objectValues.splice(0, this.objectValues.length);
            if(this.recycleBin) {
                for(object of cleared) {
                    this.recycleBin.addObject(object);
                }
            }
            this.popupVisible = true;
        },

        flush() {
            if(this.recycleBin) {
                for(object of this.objectValues) {
                    this.recycleBin.addObject(object);
                }
            }
        },

        handlePopup() {
            this.$refs.input.focus();
        },

        isClearButtonVisible() {
            if(!this.clearable) {
                return false;
            }
            if(this.objectValues && this.objectValues.length > 0) {
                return true;
            }
            return false;
        },

        handleEditorChange (event) {
            if(event.keyCode === 13) {
                this.objectValues.push({id: Quasar.utils.uid(), name: this.editedValue});
                this.editedValue = "";
                this.filteredOptions = this.options;
                event.preventDefault();
            }
        },

        handleEditorInput () {
            if(this.editedValue === "") {
                this.filteredOptions = this.options;
            }else {
                const needle = this.editedValue.toLowerCase();
                this.filteredOptions = this.options.filter(
                    v => v.toLowerCase().indexOf(needle) > -1
                );
            }
        },

        toggleSelected(option) {
            var index = this.value.indexOf(option);
            if(index >= 0) {
                this.objectValues.splice(index, 1);
            }else{
                this.objectValues.push({id: Quasar.utils.uid(), name: option});
            }
        },

        getPopupOptionStyle(option) {
            var index = this.value.indexOf(option);
            if(index < 0) {
                return "";
            } else {
                return "text-primary bg-grey-3";
            }
        },

        removeObject(object) {
            var index = this.objectValues.indexOf(object);
            if(index >= 0) {
                this.objectValues.splice(index, 1);
                if(this.recycleBin) {
                    this.recycleBin.addObject(object);
                }
            }
        },

        handleObjectEdited(object) {
            var index = this.objectValues.indexOf(object);
            this.unwatchValue = true;
            this.value.splice(index, 1, object.name);
        },

        addObject(object) {
            this.objectValues.push(object);
        },
    }
});

const app = new Vue({
    delimiters: ['[[', ']]'],
    el: '#q-app',

    data () {
        return {

            form_data: {},
            defaultFormData: {},

            submitting: false,
            confirmEdit: false,
            editButtonClicked: true,
            editButtonContact: null,

            bcPreviewSplitterFrac: 75,
            bcPreviewSelected: 1,
            bcScanUploadedFiles: [],
            bcScanMode: "Double Sided",
            bcScanModeOptions: ["One Sided", "Double Sided"],
        }
    },

    created () {
        this.form_data = context.form_data;
        this.defaultFormData = context.default_form_data;
        this.editButtonClicked = context.editButtonClicked;
        if(this.editButtonClicked) {
            this.editButtonContact = form_data.name;
        }
    },

    computed: {

        bcPreviewVirtualScrollItemSize() {
            return 144 * (100 - this.bcPreviewSplitterFrac) / 20;
        },

    },

    methods: {

        notifyServerError(response) {
            console.error("An error occured on the server. Details belows.");
            console.error(response);
            this.$q.notify({
                icon: 'error',
                color: 'negative',
                message: 'A Server error occured',
            });
        },


        uploaderFactory (files) {
            var url = this.$refs.bcScanUploaderUrl.value;
            var form_data = this.form_data;
            var scanMode = this.bcScanMode;
            return {
                url: url,
                method: 'POST',
                formFields: [
                    { name: 'scan_mode', value: scanMode },
                ]
            }
        },

        uploaderFileAdded (files) {
            for(f of files) {
                scannedFile = {
                    parent: f,
                    scanResults: this.defaultFormData,
                }
                this.bcScanUploadedFiles.push(scannedFile);
            }
            this.form_data = this.bcScanUploadedFiles[this.bcPreviewSelected-1].scanResults;
        },

        uploaderFileRemoved (files) {
            this.bcScanUploadedFiles = this.bcScanUploadedFiles.filter( f => files.indexOf(f.parent) < 0);
            if(this.bcScanUploadedFiles.length == 0) {
                this.form_data = this.defaultFormData;
            }else{
                if(this.bcPreviewSelected > this.bcScanUploadedFiles.length) {
                    this.bcPreviewSelected = this.bcScanUploadedFiles.length;
                }
                this.form_data = this.bcScanUploadedFiles[this.bcPreviewSelected-1].scanResults;
            }
        },

        bcScanFileUploaded (info) {
            response = JSON.parse(info.xhr.response);
            files = info.files;
            if(files.length < 1) {
                return;
            }
            for(file of files) {
                key = file.name;
                this.bcScanUploadedFiles = this.bcScanUploadedFiles.filter( f => f.parent.name != key);
                results = response[key];
                for(result of results) {
                    scannedFile = {
                        parent: file,
                        scanResults: result.NER,
                        hasBeenSaved: false,
                        src: result.OCR.decorated
                    }
                    this.bcScanUploadedFiles.push(scannedFile);
                }
            }
            this.form_data = this.bcScanUploadedFiles[this.bcPreviewSelected-1].scanResults;
        },

        addContact (url, editConfirmed) {
            this.submitting = true;

            allowEdit = false;
            if(editConfirmed) {
                allowEdit = true;
            }
            if(this.editButtonClicked) {
                if(this.editButtonContact == this.form_data.name) {
                    allowEdit = true;
                }
                this.editButtonClicked = false;
            }
            var self = this;
            $.ajax({
                type: "POST",
                url: url,
                data: {
                    csrfmiddlewaretoken: context.csrf_token,
                    form_data: JSON.stringify(self.form_data),
                    allow_edit: (allowEdit ? 1 : 0),
                },
                success: function callback(response) {
                    self.submitting = false;
                    if(response.status == "success") {
                        self.$q.notify({
                            icon: 'done',
                            color: 'positive',
                            message: response.message
                        });
                        self.$refs.contactForm.resetValidation();
                        if(self.bcScanUploadedFiles.length > 0) {
                            self.$set(self.bcScanUploadedFiles[self.bcPreviewSelected-1], "hasBeenSaved", true);
                            self.bcPreviewSelected = (self.bcPreviewSelected % self.bcScanUploadedFiles.length) + 1;
                        }
                    }else if(response.status == "already_exists") {
                        self.confirmEdit = true;
                    }else{
                        self.$q.notify({
                            icon: 'error',
                            color: 'negative',
                            message: response.message
                        });
                    }
                },
                error: function callback(response) {
                    self.submitting = false;
                    self.$q.notify({
                        icon: 'error',
                        color: 'negative',
                        message: 'Server Error Occured'
                    });
                }
            });

        },

        getBCPreviewThumbnailStyle (index) {
            hW = (index == this.bcPreviewSelected-1 ? "4" : "2");
            vW = (index == this.bcPreviewSelected-1 ? "4" : "1");
            style = ("width: "+this.bcPreviewVirtualScrollItemSize+"px; "+
                     "height: 100%; "+
                     "border-style: solid; "+
                     "border-width: "+hW+"px "+vW+"px "+hW+"px "+vW+"px; "+
                     "border-color: "+(index == this.bcPreviewSelected-1 ? "#8bc34a" : "gray")+";");
            return style;
        },

        bcThumbnailClicked(index) {
            this.bcPreviewSelected = index+1;
        },

        resetBCScan(uploader) {
            uploader.abort();
            uploader.reset();
            this.form_data = this.defaultFormData;
        },

        addPhone() {
            this.form_data.phone_number.push({type: "Mobile", number: []});
        },

        removePhone(index) {
            this.$refs.phones[index].flush();
            this.form_data.phone_number.splice(index, 1);
        },

    },

    watch: {

        bcPreviewSelected() {
            this.$refs.bcPreviewVirtualScroll.scrollTo(this.bcPreviewSelected-1);
            this.form_data = this.bcScanUploadedFiles[this.bcPreviewSelected-1].scanResults;
        },
    },
});
