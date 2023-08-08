import { Show, SimpleShowLayout, DateField, TextField, Datagrid, TopToolbar, Button, Resource, WithRecord, useRecordContext, useGetList } from 'react-admin'
import { DiagnosticList } from '../diagnostic/DiagnosticList'
import MemoryIcon from '@mui/icons-material/Memory'
import { API_URI, PREDICTION_API_URI } from '../config'
import { Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions } from '@mui/material'
import { useState } from 'react'

const PredictionModalContent = ({ handleClose, results, diagnosticData }: { handleClose: any, results: any, diagnosticData: any }) => {
    const tableData = results.predictions.map((result: any, idx: number) => ({
        predictedAction: result,
        ...diagnosticData[idx]
    }))

    const tableItems = [
        { 'source': 'predictedAction', label: 'Recommended Action' },
        { 'source': 'module', label: 'Module' },
        { 'source': 'faulttype', label: 'Fault Type' },
        { 'source': 'dtcfull', label: 'DTC Full' },
        { 'source': 'dtcdescription', label: 'DTC Description' }
    ]

    return <>
        <DialogTitle id="alert-dialog-title">
          Predicted Diagnostic Actions
        </DialogTitle>
        <DialogContent>
            <Datagrid
                data={tableData}
                bulkActionButtons={false}
                sort={{ field: 'id', order: 'DESC' }}
            >
                {tableItems.map(item => <TextField cellClassName='red' key={item.source} source={item.source} label={item?.label || item.source} />)}
            </Datagrid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose} label={'Close'} />
        </DialogActions>
    </>
}

const Actions = ({ currentRecord }: { currentRecord: any }) => {
    const [isOpen, setOpen] = useState<boolean>(false)
    const [diagnosticDataResults, setDiagnosticDataResults] = useState<any>(null)
    const [predictionResults, setPredictionResults] = useState<any>(null)

    return <TopToolbar>
        <Button size='large' color='primary' variant='contained' sx={{ color: '#fff' }} onClick={async () => {
            const diagnosticDataResponse = await fetch(`${API_URI}/vehicles?offset=0&limit=100&anonymised_vin=eq.${currentRecord.anonymised_vin}&sessiontimestamp=eq.${new Date(currentRecord.consultation_date).toISOString().slice(0, 10)}&order=id.asc`, {
                headers: {
                    accept: 'application/json'
                }
            })
            const diagnosticDataBody = await diagnosticDataResponse.json()

            const predictionResponse = await fetch(`${PREDICTION_API_URI}/predict`, {
                headers: {
                    accept: 'application/json',
                    'Content-Type': 'application/json'
                },
                method: 'POST',
                body: JSON.stringify(diagnosticDataBody)
            })
            const predictionBody = await predictionResponse.json()

            setOpen(true)
            setDiagnosticDataResults(diagnosticDataBody)
            setPredictionResults(predictionBody)
        }} label='Predict Diagnostic Actions'>
            <MemoryIcon />
        </Button>

        <Dialog
            open={isOpen}
            aria-labelledby="alert-dialog-title"
            aria-describedby="alert-dialog-description"
        >
            <PredictionModalContent results={predictionResults} diagnosticData={diagnosticDataResults} handleClose={() => {
                setOpen(false)
                setPredictionResults(null)
                setDiagnosticDataResults(null)
            }} />
        </Dialog>
    </TopToolbar>
}

export const ConsultationShow = () => {
    const currentRecord = useRecordContext()

    return <Show actions={<Actions currentRecord={currentRecord} />}>
        <SimpleShowLayout>
            <DateField source='consultation_date' />
            <TextField source='anonymised_vin' />
            <WithRecord label='Diagnostic Data' render={record => (
                <Resource
                    name='vehicles'
                    list={<DiagnosticList filter={{
                        anonymised_vin: record.anonymised_vin,
                        sessiontimestamp: new Date(record.consultation_date).toISOString().slice(0, 10)
                    }} items={[
                        { 'source': 'module', label: 'Module' },
                        { 'source': 'faulttype', label: 'Fault Type' },
                        { 'source': 'dtcfull', label: 'DTC Full' },
                        { 'source': 'dtcdescription', label: 'DTC Description' },
                        { 'source': 'odomiles', label: 'Odometer Reading' },
                        { 'source': 'builddate', label: 'Build Date' },
                        { 'source': 'warrantydate', label: 'Warranty Start Date' },
                    ]} />}
                    options={{
                        label: ''
                    }}
                />
            )} />
        </SimpleShowLayout>
    </Show>
}